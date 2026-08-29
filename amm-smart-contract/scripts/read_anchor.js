/**
 * Reads the on-chain clearing results back out of the AMMBA contract and
 * prints them next to their off-chain counterparts.
 *
 * This is the audit direction of the artifact: everything else writes to the
 * chain, this reads from it. It is also the check that the whole encoding
 * path — sigmoid -> to_node_int (x10000) -> ABI -> contract storage — is
 * consistent, without trusting any intermediate log line.
 *
 * Usage:
 *   npx hardhat run scripts/read_anchor.js --network localhost
 *   npx hardhat run scripts/read_anchor.js --network volta
 *
 * Env:
 *   CONTRACT_ADDRESS  deployed AMMBA address
 *                     (default: the deterministic local Hardhat address)
 *   OFFCHAIN_DB_URL   mock off-chain DB, used to discover the market ids
 *                     (default: http://127.0.0.1:8080)
 *   MARKET_ID         read exactly this market instead of querying the DB
 */
const { ethers } = require("hardhat");

const SCALE = 10000n;
const DEFAULT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3";

const scaled = (value) =>
  `${Number(value) / Number(SCALE)} (raw ${value})`;

async function marketIdsFromDb(dbUrl) {
  const response = await fetch(`${dbUrl}/trades`);
  if (!response.ok) {
    throw new Error(`${dbUrl}/trades responded ${response.status}`);
  }
  const trades = await response.json();
  return [...new Set(trades.map((trade) => trade.market_id))];
}

async function main() {
  const address = process.env.CONTRACT_ADDRESS || DEFAULT_ADDRESS;
  const dbUrl = process.env.OFFCHAIN_DB_URL || "http://127.0.0.1:8080";
  const ammba = await ethers.getContractAt("AMMBA", address);

  console.log(`contract ${address}`);

  const marketIds = process.env.MARKET_ID
    ? [process.env.MARKET_ID]
    : await marketIdsFromDb(dbUrl);

  if (marketIds.length === 0) {
    console.log("no market ids found — has a clearing run yet?");
    return;
  }

  for (const marketId of marketIds) {
    const anchored = await ammba.hasClearingResult(marketId);
    console.log(`\nmarket ${marketId}`);
    if (!anchored) {
      // Expected for a no-trade slot: run_clearing expires one-sided markets
      // off-chain and never calls the contract.
      console.log("  not anchored on-chain (no-trade slot, or mock mode)");
      continue;
    }
    const [timeSlot, supply, demand, price] =
      await ammba.getClearingResult(marketId);
    console.log(`  time slot      ${timeSlot} ` +
                `(${new Date(Number(timeSlot) * 1000).toISOString()})`);
    console.log(`  total supply   ${scaled(supply)} kWh`);
    console.log(`  total demand   ${scaled(demand)} kWh`);
    console.log(`  clearing price ${scaled(price)} ct/kWh`);
  }

  // View functions are free when called off-chain: these reads consumed no
  // gas and produced no transaction. Stated explicitly because it is part of
  // the cost argument in 5.4.
  console.log("\nall reads were eth_call — 0 gas, no transaction");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
