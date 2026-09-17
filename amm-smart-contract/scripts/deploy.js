/**
 * Deploys the AMMBA contract, configures the default community and
 * authorizes the Clearing Node address.
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network localhost   (hardhat node)
 *   npx hardhat run scripts/deploy.js --network volta       (EW testnet)
 *
 * Env:
 *   CLEARING_NODE_ADDRESS  address to whitelist (defaults to the deployer)
 *   COMMUNITY_UUID         community id string  (default "communityid_1")
 */
const { ethers } = require("hardhat");

// IMPORTANT: must match the Clearing Node's string_to_bytes32() conversion
// (keccak256 of the UTF-8 string unless the value is already a 32-byte hex).
function toBytes32(value) {
  if (/^(0x)?[0-9a-fA-F]{64}$/.test(value)) {
    return value.startsWith("0x") ? value : "0x" + value;
  }
  return ethers.keccak256(ethers.toUtf8Bytes(value));
}

// Defaults from amm-clearing-node/configuration.yaml, scaled by 10000.
const SCALE = 10000;
const COMMUNITY_DEFAULTS = {
  kUpper: BigInt(Math.round(40.0 * SCALE)), // retail buy price ct/kWh
  kLower: BigInt(Math.round(8.0 * SCALE)),  // feed-in tariff ct/kWh
  theta: BigInt(Math.round(1.0 * SCALE)),
  steepness: BigInt(Math.round(0.6 * SCALE)),
};

async function main() {
  const [deployer] = await ethers.getSigners();
  const clearingNode = process.env.CLEARING_NODE_ADDRESS || deployer.address;
  const communityUuid = process.env.COMMUNITY_UUID || "communityid_1";

  console.log(`Deployer:            ${deployer.address}`);
  console.log(`Clearing node:       ${clearingNode}`);
  console.log(`Community uuid:      ${communityUuid}`);

  // Transaction hash, block and gas are printed for every write: on a public
  // network these three lines are the deployment evidence, and reconstructing
  // them from a block explorer afterwards is avoidable work.
  const report = (label, receipt, hash) =>
    console.log(`  ${label.padEnd(20)} tx ${hash} ` +
                `block ${receipt.blockNumber} gas ${receipt.gasUsed}`);

  const AMMBA = await ethers.getContractFactory("AMMBA");
  const ammba = await AMMBA.deploy();
  await ammba.waitForDeployment();
  const address = await ammba.getAddress();
  const deployTx = ammba.deploymentTransaction();
  const deployReceipt = await deployTx.wait();
  console.log(`AMMBA deployed at:   ${address}`);
  report("deployment", deployReceipt, deployTx.hash);

  const nodeTx = await ammba.setClearingNode(clearingNode, true);
  const nodeReceipt = await nodeTx.wait();
  console.log("Clearing node authorized.");
  report("setClearingNode", nodeReceipt, nodeTx.hash);

  const communityBytes32 = toBytes32(communityUuid);
  const paramsTx = await ammba.setCommunityParams(
    communityBytes32,
    COMMUNITY_DEFAULTS.kUpper,
    COMMUNITY_DEFAULTS.kLower,
    COMMUNITY_DEFAULTS.theta,
    COMMUNITY_DEFAULTS.steepness
  );
  const paramsReceipt = await paramsTx.wait();
  console.log(`Community params set for ${communityBytes32}`);
  report("setCommunityParams", paramsReceipt, paramsTx.hash);

  // Read the parameters back out of the contract rather than trusting that
  // the write went through: the Clearing Node refuses to start if these
  // differ from configuration.yaml, so a mismatch is better found here.
  const onChain = await ammba.getCommunityParams(communityBytes32);
  console.log(`On-chain params:     k_upper ${onChain[0]}, k_lower ${onChain[1]}, ` +
              `theta ${onChain[2]}, steepness ${onChain[3]}`);
  console.log(`Owner:               ${await ammba.owner()}`);
  console.log(`Whitelisted ${clearingNode}: ` +
              `${await ammba.authorizedClearingNodes(clearingNode)}`);
  console.log(`Whitelisted ${deployer.address}: ` +
              `${await ammba.authorizedClearingNodes(deployer.address)}`);

  console.log("\nClearing Node env vars for live mode:");
  console.log(`  BLOCKCHAIN_MODE=live`);
  console.log(`  CONTRACT_ADDRESS=${address}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
