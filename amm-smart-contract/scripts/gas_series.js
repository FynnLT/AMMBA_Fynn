// Scripted clearMarket gas series (T-07), on the in-process Hardhat network.
//
//   npx hardhat run scripts/gas_series.js
//
// Compiles through Hardhat (`getContractFactory("AMMBA")`), deploys, whitelists
// a clearing node and sets the calibrated band, then records two series:
//
//   trade   20 clearings on the trade path, at the aggregates and prices of
//           the first 20 cleared slots of a recorded campaign run;
//   n_axis  clearings at the aggregates of N = 6 .. 3200 participants.
//           `clearMarket` takes aggregates only, so the expected line is flat.
//
// Market ids are keccak256 of a label: deterministic, but byte-random like a
// real id. `gas_pilot.js` used padded short strings, which is exactly the
// zero-byte case the series must avoid -- every zero byte of calldata costs
// 12 gas less than a non-zero one (D-40). `gas_pilot.js` stays as the record
// of the pilot.
//
// Output: `#` header lines, then CSV on stdout, then one `#` summary line per
// series. The Volta record is 216,843 gas on the trade path, with a spread of
// 0 or +-12 gas per zero byte in the calldata (D-40); a deviation is reported,
// not adjusted.
//
// Refuses to run on any network but the in-process `hardhat` one.
const { execSync } = require("child_process");
const hre = require("hardhat");
const { ethers, network } = hre;

// Copied from scripts/deploy.js, not imported: requiring deploy.js would run
// its `main`. Must match the Clearing Node's string_to_bytes32() conversion.
function toBytes32(value) {
  if (/^(0x)?[0-9a-fA-F]{64}$/.test(value)) {
    return value.startsWith("0x") ? value : "0x" + value;
  }
  return ethers.keccak256(ethers.toUtf8Bytes(value));
}

const SCALE = 10000;
const scaled = (x) => BigInt(Math.round(x * SCALE));

// The calibrated band (T-19, D-78), as deploy.js and configuration.yaml carry it.
const BAND = { kUpper: scaled(40.0), kLower: scaled(8.0),
               theta: scaled(1.0), steepness: scaled(0.6) };
const COMMUNITY = "communityid_1";

// The first 20 cleared slots of baseline_pro_rata--seed-0_slots.csv
// (total_supply_kwh, total_demand_kwh, clearing_price_ct), x 10,000 and
// rounded. Run baseline_pro_rata--seed-0 at repo_sha
// 703b20fc26de5465fab7f4e821b87d2d28cab33d.
const TRADE_SLOTS = [
  [1160, 193998, 286347],     // slot 1756973700
  [11157, 187848, 283988],    // slot 1756974600
  [24962, 137503, 278515],    // slot 1756975500
  [23563, 126210, 278282],    // slot 1756976400
  [51659, 89961, 260326],     // slot 1756977300
  [36384, 127169, 273752],    // slot 1756978200
  [40916, 157225, 274937],    // slot 1756979100
  [79621, 119799, 256044],    // slot 1756980000
  [110207, 100775, 235509],   // slot 1756980900
  [164533, 98696, 208402],    // slot 1756981800
  [202084, 86566, 179162],    // slot 1756982700
  [266286, 100683, 166890],   // slot 1756983600
  [357909, 74562, 109691],    // slot 1756984500
  [386405, 117990, 145100],   // slot 1756985400
  [393721, 99041, 125972],    // slot 1756986300
  [355954, 103299, 139942],   // slot 1756987200
  [354077, 134858, 167622],   // slot 1756988100
  [434405, 92993, 111839],    // slot 1756989000
  [411403, 93971, 117254],    // slot 1756989900
  [479925, 84881, 98475],     // slot 1756990800
];

const N_AXIS = [6, 50, 100, 250, 1000, 3200];
const N_AXIS_PRICE_CT = 22.8;
const TIME_SLOT_BASE = 1_757_000_000;
const SLOT_SEC = 900;

function zeroBytes(hexData) {
  const bytes = hexData.startsWith("0x") ? hexData.slice(2) : hexData;
  let zeros = 0;
  for (let i = 0; i < bytes.length; i += 2) {
    if (bytes.slice(i, i + 2) === "00") zeros += 1;
  }
  return zeros;
}

function repoState() {
  try {
    const sha = execSync("git rev-parse HEAD", { cwd: __dirname }).toString().trim();
    const dirty = execSync("git status --porcelain", { cwd: __dirname })
      .toString().trim() !== "";
    return `${sha}${dirty ? " (dirty)" : ""}`;
  } catch (error) {
    return "unknown";
  }
}

async function clear(contract, node, row) {
  const tx = await contract.connect(node).clearMarket(
    row.marketId, row.community, row.timeSlot,
    row.totalSupply, row.totalDemand, row.clearingPrice);
  const receipt = await tx.wait();
  return { ...row, zeroBytes: zeroBytes(tx.data), gasUsed: receipt.gasUsed };
}

function summary(series, rows) {
  const gas = rows.map((r) => Number(r.gasUsed));
  const mean = gas.reduce((a, b) => a + b, 0) / gas.length;
  return `# summary,series=${series},n=${gas.length},mean=${mean.toFixed(1)},` +
         `min=${Math.min(...gas)},max=${Math.max(...gas)}`;
}

async function main() {
  if (network.name !== "hardhat") {
    throw new Error(`gas_series.js runs on the in-process hardhat network ` +
                    `only, not on '${network.name}'`);
  }
  const [owner, node] = await ethers.getSigners();
  const AMMBA = await ethers.getContractFactory("AMMBA");
  const contract = await AMMBA.deploy();
  await contract.waitForDeployment();
  await (await contract.setClearingNode(node.address, true)).wait();
  const community = toBytes32(COMMUNITY);
  await (await contract.setCommunityParams(
    community, BAND.kUpper, BAND.kLower, BAND.theta, BAND.steepness)).wait();

  const buildInfo = await hre.artifacts.getBuildInfo("contracts/AMMBA.sol:AMMBA");
  console.log(`# repo_sha ${repoState()}`);
  console.log(`# hardhat ${require("hardhat/package.json").version}`);
  console.log(`# solc ${buildInfo ? buildInfo.solcLongVersion : "unknown"}`);
  console.log(`# network ${network.name}, owner ${owner.address}, ` +
              `clearing node ${node.address}`);
  console.log(`# community ${COMMUNITY} -> ${community}`);
  console.log(`# trade: first 20 cleared slots of baseline_pro_rata--seed-0; ` +
              `n_axis: i is N, supply = 0.9 N kWh, demand = N kWh, ` +
              `price ${N_AXIS_PRICE_CT} ct`);
  console.log(`# Volta reference: 216843 gas on the trade path (D-40)`);

  const trade = [];
  for (let i = 0; i < TRADE_SLOTS.length; i += 1) {
    const [supply, demand, price] = TRADE_SLOTS[i];
    trade.push(await clear(contract, node, {
      series: "trade", i,
      marketId: ethers.keccak256(ethers.toUtf8Bytes("gas-series-" + i)),
      community, timeSlot: BigInt(TIME_SLOT_BASE + i * SLOT_SEC),
      totalSupply: BigInt(supply), totalDemand: BigInt(demand),
      clearingPrice: BigInt(price) }));
  }

  const nAxis = [];
  for (let j = 0; j < N_AXIS.length; j += 1) {
    const n = N_AXIS[j];
    nAxis.push(await clear(contract, node, {
      series: "n_axis", i: n,
      marketId: ethers.keccak256(ethers.toUtf8Bytes("gas-n-" + n)),
      community,
      timeSlot: BigInt(TIME_SLOT_BASE + (TRADE_SLOTS.length + j) * SLOT_SEC),
      totalSupply: scaled(0.9 * n), totalDemand: scaled(n),
      clearingPrice: scaled(N_AXIS_PRICE_CT) }));
  }

  console.log("series,i,market_id,total_supply,total_demand,clearing_price," +
              "calldata_zero_bytes,gas_used");
  for (const r of [...trade, ...nAxis]) {
    console.log([r.series, r.i, r.marketId, r.totalSupply, r.totalDemand,
                 r.clearingPrice, r.zeroBytes, r.gasUsed].join(","));
  }
  console.log(summary("trade", trade));
  console.log(summary("n_axis", nAxis));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
