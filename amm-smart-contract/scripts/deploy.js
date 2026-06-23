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
  kUpper: BigInt(Math.round(28.5 * SCALE)), // retail buy price ct/kWh
  kLower: BigInt(Math.round(8.0 * SCALE)),  // feed-in tariff ct/kWh
  theta: BigInt(Math.round(1.0 * SCALE)),
  steepness: BigInt(Math.round(2.5 * SCALE)),
};

async function main() {
  const [deployer] = await ethers.getSigners();
  const clearingNode = process.env.CLEARING_NODE_ADDRESS || deployer.address;
  const communityUuid = process.env.COMMUNITY_UUID || "communityid_1";

  console.log(`Deployer:            ${deployer.address}`);
  console.log(`Clearing node:       ${clearingNode}`);
  console.log(`Community uuid:      ${communityUuid}`);

  const AMMBA = await ethers.getContractFactory("AMMBA");
  const ammba = await AMMBA.deploy();
  await ammba.waitForDeployment();
  const address = await ammba.getAddress();
  console.log(`AMMBA deployed at:   ${address}`);

  await (await ammba.setClearingNode(clearingNode, true)).wait();
  console.log("Clearing node authorized.");

  const communityBytes32 = toBytes32(communityUuid);
  await (
    await ammba.setCommunityParams(
      communityBytes32,
      COMMUNITY_DEFAULTS.kUpper,
      COMMUNITY_DEFAULTS.kLower,
      COMMUNITY_DEFAULTS.theta,
      COMMUNITY_DEFAULTS.steepness
    )
  ).wait();
  console.log(`Community params set for ${communityBytes32}`);

  console.log("\nClearing Node env vars for live mode:");
  console.log(`  BLOCKCHAIN_MODE=live`);
  console.log(`  CONTRACT_ADDRESS=${address}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
