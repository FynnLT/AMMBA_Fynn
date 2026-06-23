const { loadFixture } = require("@nomicfoundation/hardhat-network-helpers");
const { expect } = require("chai");
const { ethers } = require("hardhat");

// All on-chain values are scaled by 10000 (NODE_FLOAT_SCALING_FACTOR).
const SCALE = 10000n;
const K_UPPER = 285000n; // 28.5 ct/kWh
const K_LOWER = 80000n;  //  8.0 ct/kWh
const THETA = 10000n;    //  1.0
const STEEPNESS = 25000n; // 2.5

const bytes32 = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));
const MARKET = bytes32("market1");
const COMMUNITY = bytes32("communityid_1");

describe("AMMBA", function () {
  async function deployFixture() {
    const [owner, clearingNode, stranger] = await ethers.getSigners();
    const AMMBA = await ethers.getContractFactory("AMMBA");
    const ammba = await AMMBA.deploy();
    await ammba.setClearingNode(clearingNode.address, true);
    await ammba.setCommunityParams(COMMUNITY, K_UPPER, K_LOWER, THETA, STEEPNESS);
    return { ammba, owner, clearingNode, stranger };
  }

  describe("access control", function () {
    it("only the owner can set community params", async function () {
      const { ammba, stranger } = await loadFixture(deployFixture);
      await expect(
        ammba.connect(stranger).setCommunityParams(COMMUNITY, K_UPPER, K_LOWER, THETA, STEEPNESS)
      ).to.be.revertedWith("AMMBA: caller is not the owner");
    });

    it("only the owner can authorize clearing nodes", async function () {
      const { ammba, stranger } = await loadFixture(deployFixture);
      await expect(
        ammba.connect(stranger).setClearingNode(stranger.address, true)
      ).to.be.revertedWith("AMMBA: caller is not the owner");
    });

    it("rejects clearMarket from non-whitelisted callers", async function () {
      const { ammba, stranger } = await loadFixture(deployFixture);
      await expect(
        ammba.connect(stranger).clearMarket(MARKET, COMMUNITY, 123123, 125000n, 100000n, 151472n)
      ).to.be.revertedWith("AMMBA: caller is not an authorized clearing node");
    });

    it("revoked clearing nodes can no longer clear", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      await ammba.setClearingNode(clearingNode.address, false);
      await expect(
        ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, 1n, 1n, K_LOWER)
      ).to.be.revertedWith("AMMBA: caller is not an authorized clearing node");
    });
  });

  describe("community params", function () {
    it("stores and returns params", async function () {
      const { ammba } = await loadFixture(deployFixture);
      const [kU, kL, theta, b] = await ammba.getCommunityParams(COMMUNITY);
      expect([kU, kL, theta, b]).to.deep.equal([K_UPPER, K_LOWER, THETA, STEEPNESS]);
    });

    it("rejects k_upper < k_lower", async function () {
      const { ammba } = await loadFixture(deployFixture);
      await expect(
        ammba.setCommunityParams(COMMUNITY, K_LOWER, K_UPPER, THETA, STEEPNESS)
      ).to.be.revertedWith("AMMBA: k_upper below k_lower");
    });
  });

  describe("clearMarket", function () {
    // Example from the implementation guide: supply 12.5 kWh, demand 10 kWh,
    // ratio 1.25 -> sigmoid price ~15.147 ct/kWh -> 151472 scaled.
    const SUPPLY = 125000n;
    const DEMAND = 100000n;
    const PRICE = 151472n;

    it("records the result and emits MarketCleared", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      const tx = await ammba
        .connect(clearingNode)
        .clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, PRICE);
      const receipt = await tx.wait();
      const block = await ethers.provider.getBlock(receipt.blockNumber);

      await expect(tx)
        .to.emit(ammba, "MarketCleared")
        .withArgs(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, PRICE, block.timestamp);

      const [slot, supply, demand, price] = await ammba.getClearingResult(MARKET);
      expect([slot, supply, demand, price]).to.deep.equal([123123n, SUPPLY, DEMAND, PRICE]);
      expect(await ammba.hasClearingResult(MARKET)).to.equal(true);
    });

    it("returns true on success (staticCall)", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      const result = await ammba
        .connect(clearingNode)
        .clearMarket.staticCall(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, PRICE);
      expect(result).to.equal(true);
    });

    it("is idempotent: a market can only be cleared once", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      await ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, PRICE);
      await expect(
        ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, PRICE)
      ).to.be.revertedWith("AMMBA: market already cleared");
    });

    it("emits NoTrade and returns false when supply is zero", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      const result = await ammba
        .connect(clearingNode)
        .clearMarket.staticCall(MARKET, COMMUNITY, 123123, 0n, DEMAND, PRICE);
      expect(result).to.equal(false);

      await expect(ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, 0n, DEMAND, PRICE))
        .to.emit(ammba, "NoTrade")
        .withArgs(MARKET, COMMUNITY, 123123, 0n, DEMAND);
      expect(await ammba.hasClearingResult(MARKET)).to.equal(false);
    });

    it("emits NoTrade when demand is zero", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      await expect(ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, 0n, PRICE))
        .to.emit(ammba, "NoTrade");
    });

    it("rejects prices outside [K_lower, K_upper]", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      await expect(
        ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, K_UPPER + 1n)
      ).to.be.revertedWith("AMMBA: clearing price out of bounds");
      await expect(
        ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, K_LOWER - 1n)
      ).to.be.revertedWith("AMMBA: clearing price out of bounds");
    });

    it("accepts prices exactly at the bounds", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      await ammba.connect(clearingNode).clearMarket(MARKET, COMMUNITY, 123123, SUPPLY, DEMAND, K_LOWER);
      await ammba.connect(clearingNode).clearMarket(bytes32("market2"), COMMUNITY, 123123, SUPPLY, DEMAND, K_UPPER);
    });

    it("rejects clearing for communities without params", async function () {
      const { ammba, clearingNode } = await loadFixture(deployFixture);
      await expect(
        ammba.connect(clearingNode).clearMarket(MARKET, bytes32("unknown"), 123123, SUPPLY, DEMAND, PRICE)
      ).to.be.revertedWith("AMMBA: community params not set");
    });
  });

  describe("views", function () {
    it("getClearingResult reverts for unknown markets", async function () {
      const { ammba } = await loadFixture(deployFixture);
      await expect(ammba.getClearingResult(MARKET)).to.be.revertedWith("AMMBA: unknown market");
    });
  });
});
