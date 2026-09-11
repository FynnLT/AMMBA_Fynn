// Pilot gas measurement. Compiles with the npm solc (solcjs) because the
// native compiler download is blocked in this sandbox; gas units are
// identical either way.
const fs = require("fs");
const path = require("path");
const solc = require("solc");
const { ethers } = require("hardhat");

function compile() {
  const file = "AMMBA.sol";
  const source = fs.readFileSync(path.join(__dirname, "../contracts", file), "utf8");
  const input = {
    language: "Solidity",
    sources: { [file]: { content: source } },
    settings: {
      optimizer: { enabled: true, runs: 200 },
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } },
    },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  (out.errors || []).filter(e => e.severity === "error").forEach(e => { throw new Error(e.formattedMessage); });
  const c = out.contracts[file]["AMMBA"];
  return { abi: c.abi, bytecode: "0x" + c.evm.bytecode.object };
}

async function main() {
  const { abi, bytecode } = compile();
  const [owner, node] = await ethers.getSigners();
  const F = new ethers.ContractFactory(abi, bytecode, owner);
  const c = await F.deploy();
  await c.waitForDeployment();
  const rows = [];
  rows.push(["deploy", "", Number((await c.deploymentTransaction().wait()).gasUsed)]);
  rows.push(["setClearingNode", "", Number((await (await c.setClearingNode(node.address, true)).wait()).gasUsed)]);
  const cu = ethers.encodeBytes32String("pilot-community");
  rows.push(["setCommunityParams", "", Number((await (await c.setCommunityParams(cu, 285000n, 80000n, 10000n, 25000n)).wait()).gasUsed)]);

  let i = 0;
  for (const n of [6, 25, 50, 100, 200, 400, 800, 1600, 3200]) {
    const mid = ethers.encodeBytes32String("m-" + n);
    const rc = await (await c.connect(node).clearMarket(
      mid, cu, 1700000000 + (i++) * 900,
      BigInt(Math.round(n * 0.9 * 10000)), BigInt(n * 10000), 151472n)).wait();
    rows.push(["clearMarket", n, Number(rc.gasUsed)]);
  }
  const nt = await (await c.connect(node).clearMarket(
    ethers.encodeBytes32String("m-nt"), cu, 1700099999, 0n, 100000n, 151472n)).wait();
  rows.push(["clearMarket_noTrade", 0, Number(nt.gasUsed)]);

  console.log("op,participants,gas");
  rows.forEach(r => console.log(r.join(",")));
}
main().catch(e => { console.error(e); process.exit(1); });
