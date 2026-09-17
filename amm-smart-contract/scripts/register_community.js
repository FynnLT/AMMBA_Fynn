/**
 * Registers (or updates) the sigmoid parameters of one community on an already
 * deployed AMMBA contract.
 *
 * Needed because `clearMarket` opens with `require(params.exists)`, so any
 * community the contract has never seen makes every clearing revert with
 * "AMMBA: community params not set". `deploy.js` registers exactly one
 * community; this script registers any further one without redeploying.
 *
 * The demo UI derives its community id from the community *name* —
 * `community-${slug(name)}` in ui/app.js — so the default name "Community 1"
 * produces `community-community-1`, not `communityid_1`.
 *
 * Usage:
 *   npx hardhat run scripts/register_community.js --network volta
 *
 * Env:
 *   CONTRACT_ADDRESS       deployed AMMBA address (required)
 *   DEPLOYER_PRIVATE_KEY   the contract owner — setCommunityParams is onlyOwner
 *   COMMUNITY_UUID         community id string (required)
 *   K_UPPER K_LOWER THETA STEEPNESS   ct/kWh and dimensionless, unscaled
 *                          (defaults: 40.0 / 8.0 / 1.0 / 0.6, i.e. the values
 *                          in amm-clearing-node/configuration.yaml)
 */
const { ethers } = require("hardhat");

const SCALE = 10000;

// Must match the Clearing Node's string_to_bytes32() and deploy.js::toBytes32:
// keccak256 of the UTF-8 string unless the value is already a 32-byte hex.
function toBytes32(value) {
  if (/^(0x)?[0-9a-fA-F]{64}$/.test(value)) {
    return value.startsWith("0x") ? value : "0x" + value;
  }
  return ethers.keccak256(ethers.toUtf8Bytes(value));
}

const scaled = (name, fallback) =>
  BigInt(Math.round(parseFloat(process.env[name] ?? fallback) * SCALE));

function requireEnv(name) {
  const value = process.env[name];
  if (!value) throw new Error(`missing env var ${name}`);
  return value;
}

async function main() {
  const address = requireEnv("CONTRACT_ADDRESS");
  const communityUuid = requireEnv("COMMUNITY_UUID");
  const owner = new ethers.Wallet(requireEnv("DEPLOYER_PRIVATE_KEY"),
                                  ethers.provider);
  const ammba = (await ethers.getContractAt("AMMBA", address)).connect(owner);

  const onChainOwner = await ammba.owner();
  if (onChainOwner.toLowerCase() !== owner.address.toLowerCase()) {
    throw new Error(`setCommunityParams is onlyOwner: contract owner is ` +
                    `${onChainOwner}, key given is ${owner.address}`);
  }

  const id = toBytes32(communityUuid);
  const params = [scaled("K_UPPER", 40.0), scaled("K_LOWER", 8.0),
                  scaled("THETA", 1.0), scaled("STEEPNESS", 0.6)];

  console.log(`contract   ${address}`);
  console.log(`community  ${communityUuid}`);
  console.log(`bytes32    ${id}`);
  console.log(`params     k_upper ${params[0]}, k_lower ${params[1]}, ` +
              `theta ${params[2]}, steepness ${params[3]}`);

  try {
    const existing = await ammba.getCommunityParams(id);
    console.log(`already registered as ${existing.join(", ")} — overwriting`);
  } catch {
    console.log("not registered yet");
  }

  const tx = await ammba.setCommunityParams(id, ...params);
  const receipt = await tx.wait();
  console.log(`\ntx ${receipt.hash} block ${receipt.blockNumber} ` +
              `gas ${receipt.gasUsed}`);

  // Read back rather than trust the write: the Clearing Node refuses to start
  // if these differ from configuration.yaml, so a mismatch is better found now.
  console.log(`on-chain   ${(await ammba.getCommunityParams(id)).join(", ")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
