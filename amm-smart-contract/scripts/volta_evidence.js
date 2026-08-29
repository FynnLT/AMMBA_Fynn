/**
 * Produces the contract behaviours that the running system never reaches, as
 * transactions and as revert reasons.
 *
 * Three of the five cases are unreachable from the demo by design:
 *   - NoTrade      run_clearing expires one-sided markets off-chain
 *                  (_expire_one_sided_market) and never calls the contract.
 *   - unauthorized the Clearing Node is whitelisted, so it never gets rejected.
 *   - re-clearing  the node answers `already_cleared` from the off-chain trade
 *                  store before the contract is consulted, so the on-chain
 *                  replay guard is only observable from a script.
 *
 * Two artefacts are collected per failing case, because they are available
 * under different conditions:
 *
 *   1. The revert reason via eth_call (staticCall). Always available, on every
 *      network, costs nothing. This is the durable evidence.
 *   2. The mined transaction with status 0, if the network accepts a failing
 *      transaction at all. Hardhat Network does NOT: it simulates on
 *      eth_sendTransaction and rejects with the revert reason instead of
 *      mining, so locally case 2 reports "not mined". A public chain such as
 *      Volta accepts the transaction and mines it with status 0, which is the
 *      hash that goes into appendix A.
 *
 * Run against the local Hardhat node FIRST, then against Volta.
 *
 * Usage:
 *   npx hardhat run scripts/volta_evidence.js --network localhost
 *   npx hardhat run scripts/volta_evidence.js --network volta
 *
 * Env:
 *   CONTRACT_ADDRESS           deployed AMMBA address (required)
 *   DEPLOYER_PRIVATE_KEY       key A — contract owner, NOT whitelisted
 *   CLEARING_NODE_PRIVATE_KEY  key B — the whitelisted clearing node
 *   COMMUNITY_UUID             default "communityid_1"
 *   EVIDENCE_CSV               output file (default volta_evidence.csv)
 */
const fs = require("fs");
const { ethers } = require("hardhat");

// Aggregates from the guide's reference example, scaled by 10000:
// 12.5 kWh supply, 10 kWh demand, 15.1472 ct/kWh.
const SUPPLY = 125000n;
const DEMAND = 100000n;
const PRICE = 151472n;

// A revert must not be gas-estimated first: ethers runs eth_estimateGas before
// sending, the estimate fails for a reverting call, and the error is thrown
// locally without a transaction ever being sent. An explicit limit forces the
// send on networks that accept failing transactions.
const REVERT_GAS = { gasLimit: 200000 };

const randomId = () => ethers.hexlify(ethers.randomBytes(32));

function requireEnv(name) {
  const value = process.env[name];
  if (!value) throw new Error(`missing env var ${name}`);
  return value;
}

/**
 * Pull the Solidity revert string out of whatever the provider threw.
 *
 * Hardhat Network decodes it for us. Volta does not: its RPC answers a
 * reverting eth_call with a bare "VM execution error" and, depending on the
 * client, either no data at all or the raw Error(string) payload. Both are
 * handled; an empty result is not an error, only a missing convenience.
 */
function revertReason(error) {
  if (error.reason) return error.reason;

  // ABI-encoded Error(string): selector 0x08c379a0 + offset + length + bytes.
  const data = error.data ?? error.error?.data ?? error.info?.error?.data ??
               error.data?.data;
  if (typeof data === "string" && data.startsWith("0x08c379a0")) {
    try {
      const [decoded] = ethers.AbiCoder.defaultAbiCoder()
        .decode(["string"], `0x${data.slice(10)}`);
      return decoded;
    } catch {
      // fall through to the message scan
    }
  }

  const message = `${error.message ?? ""} ${error.shortMessage ?? ""}`;
  const match = message.match(/reverted with reason string '([^']*)'/) ||
                message.match(/reverted with custom error '([^']*)'/) ||
                message.match(/reverted: ([^\n"]*)/);
  return match ? match[1].trim() : "";
}

/**
 * Wait for a receipt by polling, instead of TransactionResponse.wait().
 *
 * ethers' wait() throws for a status-0 receipt and tries to reconstruct the
 * revert reason by replaying the call first — which fails on an RPC that does
 * not return revert data, losing the receipt with it. The receipt is the
 * evidence, so it is fetched directly.
 */
async function waitForReceipt(hash, timeoutMs = 180000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const receipt = await ethers.provider.getTransactionReceipt(hash);
    if (receipt) return receipt;
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
  throw new Error(`no receipt for ${hash} after ${timeoutMs / 1000}s`);
}

async function main() {
  const address = requireEnv("CONTRACT_ADDRESS");
  const owner = new ethers.Wallet(requireEnv("DEPLOYER_PRIVATE_KEY"),
                                  ethers.provider);
  const node = new ethers.Wallet(requireEnv("CLEARING_NODE_PRIVATE_KEY"),
                                 ethers.provider);
  const communityUuid = process.env.COMMUNITY_UUID || "communityid_1";
  const community = ethers.keccak256(ethers.toUtf8Bytes(communityUuid));
  const ammba = await ethers.getContractAt("AMMBA", address);

  const network = await ethers.provider.getNetwork();
  console.log(`network      chainId ${network.chainId}`);
  console.log(`contract     ${address}`);
  console.log(`community    ${communityUuid} -> ${community}`);
  console.log(`owner    (A) ${owner.address}`);
  console.log(`node     (B) ${node.address}\n`);

  // Fail early and clearly rather than half way through the run.
  if (!(await ammba.authorizedClearingNodes(node.address))) {
    throw new Error(`${node.address} is not whitelisted — wrong key or ` +
                    `wrong contract`);
  }
  if (await ammba.authorizedClearingNodes(owner.address)) {
    throw new Error(`${owner.address} is whitelisted; the rejection case ` +
                    `needs an unauthorised owner — redeploy with ` +
                    `CLEARING_NODE_ADDRESS set to B`);
  }

  const rows = [];

  const record = (row) => {
    rows.push(row);
    console.log(`${row.case.padEnd(20)} ${row.status.padEnd(10)} ` +
                `gas ${String(row.gas_used).padStart(7)}  ` +
                `${row.tx_hash}${row.reason ? `  — ${row.reason}` : ""}`);
  };

  /**
   * @param label     case name
   * @param signer    which wallet sends it
   * @param args      clearMarket arguments
   * @param expected  "success" or "reverted"
   */
  const run = async (label, signer, args, expected) => {
    const contract = ammba.connect(signer);

    // Step 1: eth_call. Free, no transaction. Best effort only — some RPCs
    // return the revert reason, some do not, and a missing reason must not
    // stop the run.
    let reason = "";
    let callSucceeded = false;
    try {
      await contract.clearMarket.staticCall(...args);
      callSucceeded = true;
    } catch (error) {
      if (expected === "success") throw error;
      reason = revertReason(error) || "(not returned by this RPC)";
    }
    if (callSucceeded && expected === "reverted") {
      throw new Error(`${label}: expected a revert, the eth_call succeeded`);
    }

    // Step 2: send it for real and fetch the receipt by polling.
    let response;
    try {
      response = await contract.clearMarket(
        ...args, ...(expected === "reverted" ? [REVERT_GAS] : []));
    } catch (error) {
      // The network refused to accept a failing transaction at submission
      // (Hardhat Network does this; some public RPCs do too). Expected
      // locally, and not fatal: the revert reason above is the evidence.
      const rejected = revertReason(error);
      if (expected === "reverted") {
        record({
          case: label,
          status: "reverted",
          tx_hash: "not mined (rejected at submission)",
          block: "",
          gas_used: "",
          sender: signer.address,
          reason: reason.startsWith("(") ? rejected || reason : reason,
        });
        return;
      }
      throw error;
    }

    const receipt = await waitForReceipt(response.hash);
    const status = receipt.status === 1 ? "success" : "reverted";
    if (status !== expected) {
      throw new Error(`${label}: expected ${expected}, got ${status} ` +
                      `in ${receipt.hash}`);
    }
    record({
      case: label,
      status,
      tx_hash: receipt.hash,
      block: receipt.blockNumber.toString(),
      gas_used: receipt.gasUsed.toString(),
      sender: receipt.from,
      reason,
    });
  };

  // 1) NoTrade: authorised caller, supply side empty. Cheaper branch, no
  //    storage write, emits NoTrade and returns false without reverting.
  await run("no_trade", node,
            [randomId(), community, 1787994000, 0n, DEMAND, 0n], "success");

  // 2) Whitelist rejection by the CONTRACT OWNER. The stronger claim: not
  //    "some address is rejected" but "not even the owner may write a result".
  await run("unauthorized", owner,
            [randomId(), community, 1787994900, SUPPLY, DEMAND, PRICE],
            "reverted");

  // 3) A normal clearing, as the baseline for the comparison.
  const marketId = randomId();
  await run("clear_ok", node,
            [marketId, community, 1787995800, SUPPLY, DEMAND, PRICE],
            "success");

  // 4) The same market again: on-chain replay protection, a second and
  //    independent layer behind the node's off-chain `already_cleared`.
  await run("already_cleared", node,
            [marketId, community, 1787995800, SUPPLY, DEMAND, PRICE],
            "reverted");

  // 5) A price outside [k_lower, k_upper] from the authorised node: the bounds
  //    check is the contract's only economic assertion, so it is worth showing
  //    that it bites.
  await run("price_out_of_bounds", node,
            [randomId(), community, 1787996700, SUPPLY, DEMAND, 999999n],
            "reverted");

  const csvPath = process.env.EVIDENCE_CSV || "volta_evidence.csv";
  const columns = ["case", "status", "tx_hash", "block", "gas_used",
                   "sender", "reason"];
  const body = rows
    .map((row) => columns.map((key) => `"${row[key] ?? ""}"`).join(","))
    .join("\n");
  fs.writeFileSync(csvPath, `${columns.join(",")}\n${body}\n`);
  console.log(`\n${rows.length} cases written to ${csvPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
