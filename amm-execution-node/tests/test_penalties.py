import pytest

from src.penalties import (BALANCED, DEMAND_LIMITED, SUPPLY_LIMITED,
                           buyer_externality_penalty, determine_round_type,
                           seller_externality_penalty,
                           seller_shortfall_penalty)
from src.sigmoid import sigmoid_price

SIGMOID = {"k_upper": 28.5, "k_lower": 8.0, "theta": 1.0, "steepness": 2.5}


def test_round_type():
    assert determine_round_type(8.0, 10.0) == SUPPLY_LIMITED
    assert determine_round_type(12.5, 10.0) == DEMAND_LIMITED
    assert determine_round_type(10.0, 10.0) == BALANCED


class TestSellerShortfall:
    def test_basic_formula(self):
        # Φ = γ * K_upper * max(0, traded - delivered - η)
        result = seller_shortfall_penalty(4.0, 3.0, k_upper=28.5,
                                          gamma=1.1, eta=0.0)
        assert result["k_sho_ct_per_kwh"] == pytest.approx(31.35)
        assert result["shortfall_kwh"] == pytest.approx(1.0)
        assert result["penalty_ct"] == pytest.approx(31.35)

    def test_no_penalty_when_delivered_in_full(self):
        result = seller_shortfall_penalty(4.0, 4.0, 28.5, 1.1, 0.0)
        assert result["penalty_ct"] == 0.0

    def test_overdelivery_is_not_a_shortfall(self):
        result = seller_shortfall_penalty(4.0, 5.0, 28.5, 1.1, 0.0)
        assert result["penalty_ct"] == 0.0

    def test_eta_tolerance_absorbs_small_deviations(self):
        result = seller_shortfall_penalty(4.0, 3.95, 28.5, 1.1, eta=0.1)
        assert result["penalty_ct"] == 0.0
        result = seller_shortfall_penalty(4.0, 3.5, 28.5, 1.1, eta=0.1)
        assert result["shortfall_kwh"] == pytest.approx(0.4)


class TestSellerExternality:
    # Supply-limited round: supply 8, demand 10, traded quantity 8.
    SUPPLY, DEMAND, TRADED_QTY = 8.0, 10.0, 8.0
    PRICE = sigmoid_price(SUPPLY / DEMAND, **SIGMOID)

    def _penalty(self, traded, deliverable, eta=0.0):
        return seller_externality_penalty(
            traded, deliverable, eta, self.SUPPLY, self.DEMAND,
            self.TRADED_QTY, self.PRICE, SIGMOID)

    def test_withholding_creates_counterfactual_penalty(self):
        # Seller traded 3 but could have delivered 5 -> withheld 2 kWh.
        result = self._penalty(traded=3.0, deliverable=5.0)
        assert result["withheld_kwh"] == pytest.approx(2.0)
        # counterfactual: supply 10 / demand 10 -> ratio 1.0 == theta
        assert result["counterfactual_ratio"] == pytest.approx(1.0)
        assert result["counterfactual_price_ct_per_kwh"] == pytest.approx(18.25)
        expected = (self.PRICE - 18.25) * self.TRADED_QTY
        assert result["penalty_ct"] == pytest.approx(expected, abs=1e-4)
        # independently computed reference value
        assert result["penalty_ct"] == pytest.approx(20.083, abs=2e-3)

    def test_no_penalty_without_withholding(self):
        assert self._penalty(traded=3.0, deliverable=3.0) is None
        assert self._penalty(traded=3.0, deliverable=2.0) is None  # shortfall case

    def test_eta_tolerance(self):
        assert self._penalty(traded=3.0, deliverable=3.05, eta=0.1) is None

    def test_penalty_is_never_negative(self):
        result = self._penalty(traded=3.0, deliverable=5.0)
        assert result["penalty_ct"] >= 0.0


class TestBuyerExternality:
    # Demand-limited round (guide example): supply 12.5, demand 10.
    SUPPLY, DEMAND, TRADED_QTY = 12.5, 10.0, 10.0
    PRICE = sigmoid_price(SUPPLY / DEMAND, **SIGMOID)

    def _penalty(self, reported, actual):
        return buyer_externality_penalty(
            reported, actual, self.SUPPLY, self.DEMAND, self.TRADED_QTY,
            self.PRICE, SIGMOID)

    def test_underreporting_creates_counterfactual_penalty(self):
        # Buyer reported 2.5 kWh but actually consumed 4.5 kWh.
        result = self._penalty(reported=2.5, actual=4.5)
        assert result["underreported_kwh"] == pytest.approx(2.0)
        # counterfactual: supply 12.5 / demand 12 -> higher price
        assert result["counterfactual_ratio"] == pytest.approx(12.5 / 12.0)
        p_cf = sigmoid_price(12.5 / 12.0, **SIGMOID)
        assert result["counterfactual_price_ct_per_kwh"] == pytest.approx(
            p_cf, abs=1e-6)
        assert result["penalty_ct"] == pytest.approx(
            (p_cf - self.PRICE) * self.TRADED_QTY, abs=1e-4)
        # independently computed reference value
        assert result["penalty_ct"] == pytest.approx(25.694, abs=2e-3)

    def test_no_penalty_when_consumption_matches_or_undershoots(self):
        assert self._penalty(reported=2.5, actual=2.5) is None
        assert self._penalty(reported=2.5, actual=1.0) is None
