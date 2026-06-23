// Individual plugins pinned instead of @nomicfoundation/hardhat-toolbox:
// the toolbox floats its peer plugins, which now require hardhat >=2.28
// (Node 18+). These pins keep the project working on Node 16+.
require("@nomicfoundation/hardhat-ethers");
require("@nomicfoundation/hardhat-chai-matchers");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: { enabled: true, runs: 200 },
    },
  },
  networks: {
    // `npx hardhat node` in a second terminal, then --network localhost
    localhost: {
      url: "http://127.0.0.1:8545",
    },
    // Energy Web Volta testnet (guide §3.6). Set DEPLOYER_PRIVATE_KEY first.
    volta: {
      url: process.env.VOLTA_RPC_URL || "https://volta-rpc.energyweb.org",
      accounts: process.env.DEPLOYER_PRIVATE_KEY ? [process.env.DEPLOYER_PRIVATE_KEY] : [],
    },
    // Energy Web Chain mainnet — only after Volta testing (guide §3.6).
    energyweb: {
      url: process.env.EW_RPC_URL || "https://rpc.energyweb.org",
      accounts: process.env.DEPLOYER_PRIVATE_KEY ? [process.env.DEPLOYER_PRIVATE_KEY] : [],
    },
  },
};
