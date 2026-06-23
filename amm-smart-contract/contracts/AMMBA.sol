// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title AMMBA — on-chain anchor for P2P energy market batch-auction clearing
/// @notice Receives anonymized, aggregated inputs (total supply / total demand
///         in kWh and the uniform clearing price) per market slot, verifies the
///         price against the community's configured bounds, stores the result
///         immutably and emits a `MarketCleared` event. Individual orders are
///         never submitted on-chain.
/// @dev    The sigmoid price function itself is computed OFF-CHAIN by the AMM
///         Clearing Node (recommended approach in the implementation guide,
///         §3.3): the EVM has no native floating point and an on-chain exp()
///         approximation adds cost and precision risk for no audit benefit.
///         The contract instead verifies `K_lower <= price <= K_upper` and
///         anchors all inputs needed to recompute the price off-chain.
///
///         All energy and price values are integers scaled by 10000
///         (NODE_FLOAT_SCALING_FACTOR, matching GSY-DEX):
///         0.26 kWh -> 2600, 28.5 ct/kWh -> 285000.
contract AMMBA {
    uint256 public constant SCALING_FACTOR = 10000;

    struct CommunityParams {
        uint256 kUpper;    // retail buy price, ct/kWh x 10000
        uint256 kLower;    // feed-in tariff,   ct/kWh x 10000
        uint256 theta;     // sigmoid midpoint,         x 10000
        uint256 steepness; // sigmoid steepness B,      x 10000
        bool exists;
    }

    struct ClearingResult {
        bytes32 communityUuid;
        uint256 timeSlot;
        uint256 totalSupply;   // kWh x 10000
        uint256 totalDemand;   // kWh x 10000
        uint256 clearingPrice; // ct/kWh x 10000
        address caller;
        uint256 timestamp;
        bool exists;
    }

    address public owner;
    mapping(address => bool) public authorizedClearingNodes;
    mapping(bytes32 => CommunityParams) private _communityParams;
    mapping(bytes32 => ClearingResult) private _results;

    event MarketCleared(
        bytes32 indexed market_id,
        bytes32 indexed community_uuid,
        uint256 time_slot,
        uint256 total_supply,
        uint256 total_demand,
        uint256 clearing_price,
        uint256 block_timestamp
    );

    event NoTrade(
        bytes32 indexed market_id,
        bytes32 indexed community_uuid,
        uint256 time_slot,
        uint256 total_supply,
        uint256 total_demand
    );

    event CommunityParamsSet(
        bytes32 indexed community_uuid,
        uint256 k_upper,
        uint256 k_lower,
        uint256 theta,
        uint256 steepness
    );

    event ClearingNodeSet(address indexed node, bool authorized);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "AMMBA: caller is not the owner");
        _;
    }

    modifier onlyClearingNode() {
        require(authorizedClearingNodes[msg.sender], "AMMBA: caller is not an authorized clearing node");
        _;
    }

    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    // --------------------------------------------------------------- admin

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "AMMBA: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    /// @notice Whitelist (or revoke) an AMM Clearing Node address (§3.4).
    function setClearingNode(address node, bool authorized) external onlyOwner {
        require(node != address(0), "AMMBA: node is the zero address");
        authorizedClearingNodes[node] = authorized;
        emit ClearingNodeSet(node, authorized);
    }

    /// @notice Set per-community sigmoid parameters and price bounds (§3.4).
    function setCommunityParams(
        bytes32 community_uuid,
        uint256 k_upper,
        uint256 k_lower,
        uint256 theta,
        uint256 steepness
    ) external onlyOwner {
        require(k_upper >= k_lower, "AMMBA: k_upper below k_lower");
        _communityParams[community_uuid] =
            CommunityParams(k_upper, k_lower, theta, steepness, true);
        emit CommunityParamsSet(community_uuid, k_upper, k_lower, theta, steepness);
    }

    // ------------------------------------------------------------ clearing

    /// @notice Record the clearing of one market slot.
    /// @dev    `clearing_price` is computed off-chain via the sigmoid
    ///         `price = K_upper - (K_upper - K_lower) / (1 + exp(-B*(ratio - theta)))`
    ///         with `ratio = total_supply / total_demand`. The contract verifies
    ///         the price lies within the community's `[K_lower, K_upper]` band
    ///         and anchors inputs + result for audit.
    /// @return executed true if a clearing result was recorded, false when one
    ///         side of the market was empty (no-trade event emitted instead).
    function clearMarket(
        bytes32 market_id,
        bytes32 community_uuid,
        uint256 time_slot,
        uint256 total_supply,
        uint256 total_demand,
        uint256 clearing_price
    ) external onlyClearingNode returns (bool executed) {
        CommunityParams memory params = _communityParams[community_uuid];
        require(params.exists, "AMMBA: community params not set");
        require(!_results[market_id].exists, "AMMBA: market already cleared");

        if (total_supply == 0 || total_demand == 0) {
            emit NoTrade(market_id, community_uuid, time_slot, total_supply, total_demand);
            return false;
        }

        require(
            clearing_price >= params.kLower && clearing_price <= params.kUpper,
            "AMMBA: clearing price out of bounds"
        );

        _results[market_id] = ClearingResult({
            communityUuid: community_uuid,
            timeSlot: time_slot,
            totalSupply: total_supply,
            totalDemand: total_demand,
            clearingPrice: clearing_price,
            caller: msg.sender,
            timestamp: block.timestamp,
            exists: true
        });

        emit MarketCleared(
            market_id,
            community_uuid,
            time_slot,
            total_supply,
            total_demand,
            clearing_price,
            block.timestamp
        );
        return true;
    }

    // --------------------------------------------------------------- views

    function getClearingResult(bytes32 market_id)
        external
        view
        returns (uint256 time_slot, uint256 supply, uint256 demand, uint256 price)
    {
        ClearingResult memory result = _results[market_id];
        require(result.exists, "AMMBA: unknown market");
        return (result.timeSlot, result.totalSupply, result.totalDemand, result.clearingPrice);
    }

    function hasClearingResult(bytes32 market_id) external view returns (bool) {
        return _results[market_id].exists;
    }

    function getCommunityParams(bytes32 community_uuid)
        external
        view
        returns (uint256 k_upper, uint256 k_lower, uint256 theta, uint256 steepness)
    {
        CommunityParams memory params = _communityParams[community_uuid];
        require(params.exists, "AMMBA: community params not set");
        return (params.kUpper, params.kLower, params.theta, params.steepness);
    }
}
