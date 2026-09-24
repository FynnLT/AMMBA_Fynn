import logging

import pytest

from src.penalties import (BALANCED, DEMAND_LIMITED, SUPPLY_LIMITED,
                           buyer_externality_penalty, determine_round_type,
                           redistribution, seller_externality_penalty,
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


class TestRedistribution:
    """Proportional compensation of the price damage (D-42/D-43/D-64).

    Rows are constructed directly rather than driven through `run_execution`:
    the function's contract is the participant-row shape, and hand-computed
    figures make the budget balance checkable by eye.
    """

    @staticmethod
    def _row(area, role, traded, *, externality=0.0, p_cf=None, shortfall=0.0):
        return {"area_uuid": area, "role": role, "traded_kwh": traded,
                "externality_penalty_ct": externality,
                "shortfall_penalty_ct": shortfall,
                "counterfactual_price_ct_per_kwh": p_cf}

    def test_budget_balance_is_zero_for_a_single_deviator(self):
        # p = 16.0, p_cf = 15.0, Q_t = 10 -> pool = 1.0 * 10 = 10.00 ct.
        # Buyers hold 4 and 6 kWh -> damages 4.00 and 6.00 ct.
        rows = [
            self._row("area_pv", "seller", 10.0, externality=10.0, p_cf=15.0),
            self._row("area_h1", "buyer", 4.0),
            self._row("area_h2", "buyer", 6.0),
        ]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        assert result["rule"] == "proportional"
        assert result["computed_only"] is True
        assert result["harmed_side"] == "buyer"
        assert result["penalty_pool_ct"] == pytest.approx(10.0)
        assert result["budget_balance_ct"] == pytest.approx(0.0, abs=1e-6)
        by_area = {r["area_uuid"]: r for r in result["rows"]}
        assert by_area["area_h1"]["damage_ct"] == pytest.approx(4.0)
        assert by_area["area_h1"]["compensation_ct"] == pytest.approx(4.0)
        assert by_area["area_h2"]["damage_ct"] == pytest.approx(6.0)
        assert by_area["area_h2"]["compensation_ct"] == pytest.approx(6.0)

    def test_the_profiting_side_receives_nothing(self):
        """The whole point of the D-43 correction.

        A withholding seller raises the price, so the *other sellers* profit
        from the deviation — they are not harmed and must not be compensated.
        Spreading the pool over both market sides would make the summed
        quantity 2*Q_t, cover each harmed buyer at roughly half his damage
        (4.00 ct becomes 2.35 ct here) and still report
        `budget_balance_ct == 0.00`, because the pool is fully distributed
        either way. The error would be invisible in exactly the number meant
        to prove the mechanism.
        """
        rows = [
            self._row("area_pv", "seller", 10.0, externality=10.0, p_cf=15.0),
            self._row("area_pv2", "seller", 4.0),   # profits, not harmed
            self._row("area_pv3", "seller", 3.0),   # profits, not harmed
            self._row("area_h1", "buyer", 4.0),
            self._row("area_h2", "buyer", 6.0),
        ]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        areas = {r["area_uuid"] for r in result["rows"]}
        assert areas == {"area_h1", "area_h2"}
        by_area = {r["area_uuid"]: r for r in result["rows"]}
        assert by_area["area_h1"]["compensation_ct"] == pytest.approx(4.0)
        # the figure the both-sides reading would have produced
        assert by_area["area_h1"]["compensation_ct"] != pytest.approx(
            2.35, abs=1e-2)
        assert result["budget_balance_ct"] == pytest.approx(0.0, abs=1e-6)

    def test_a_clean_round_returns_zeros_without_dividing(self):
        rows = [self._row("area_pv", "seller", 10.0),
                self._row("area_h1", "buyer", 10.0)]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        assert result["harmed_side"] is None
        assert result["penalty_pool_ct"] == 0.0
        assert result["compensated_ct"] == 0.0
        assert result["budget_balance_ct"] == 0.0
        assert result["rows"] == []
        assert result["excluded_deviators"] == []

    def test_a_seller_with_a_shortfall_is_still_compensated(self):
        """D-64: only an externality penalty excludes from the harmed set.

        Demand-limited round — a buyer underreports, so the sellers are
        harmed. One of them carries a shortfall penalty: that does not
        manipulate the price and is already penalised on its own axis, so he
        stays in the harmed set.
        """
        rows = [
            self._row("area_h1", "buyer", 10.0, externality=10.0, p_cf=17.0),
            self._row("area_pv", "seller", 6.0, shortfall=31.35),
            self._row("area_bat", "seller", 4.0),
        ]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        assert result["harmed_side"] == "seller"
        by_area = {r["area_uuid"]: r for r in result["rows"]}
        assert "area_pv" in by_area
        assert by_area["area_pv"]["compensation_ct"] == pytest.approx(6.0)
        assert by_area["area_bat"]["compensation_ct"] == pytest.approx(4.0)
        assert result["budget_balance_ct"] == pytest.approx(0.0, abs=1e-6)

    def test_several_deviators_report_a_balance_without_claiming_zero(self):
        # Two deviating sellers. Counterfactuals are not additive, so the
        # budget balance need not be zero; the point is that the number is
        # visible in the output rather than averaged away.
        rows = [
            self._row("area_pva", "seller", 6.0, externality=10.0, p_cf=15.0),
            self._row("area_pvb", "seller", 4.0, externality=5.0, p_cf=15.5),
            self._row("area_h1", "buyer", 4.0),
            self._row("area_h2", "buyer", 6.0),
        ]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        assert result["penalty_pool_ct"] == pytest.approx(15.0)
        assert isinstance(result["budget_balance_ct"], float)
        assert set(result["excluded_deviators"]) == {"area_pva", "area_pvb"}
        assert {r["area_uuid"] for r in result["rows"]} == {"area_h1", "area_h2"}

    def test_a_deviator_is_excluded_from_the_harmed_set(self):
        """Exercises the D-43 exclusion rule against constructed rows.

        The present round model cannot trigger it: externality penalties are
        applied on one market side only (`round_kind` is SUPPLY_LIMITED or
        DEMAND_LIMITED) and the harmed set is the opposite side, so a
        deviator can never be in the harmed set. The rule is implemented for
        the case the model does not produce; a market scenario for it would
        be a scenario this artifact cannot generate.
        """
        rows = [
            self._row("area_pv", "seller", 10.0, externality=10.0, p_cf=15.0),
            # Constructed: a buyer that also carries an externality penalty.
            self._row("area_h1", "buyer", 4.0, externality=2.0, p_cf=17.0),
            self._row("area_h2", "buyer", 6.0),
        ]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        areas = {r["area_uuid"] for r in result["rows"]}
        assert areas == {"area_h2"}
        assert "area_h1" in result["excluded_deviators"]

    def test_the_harmed_side_sums_to_the_traded_quantity(self, caplog):
        # The +-0 case above: one deviating seller, buyers holding 4 and 6 kWh
        # against Q_t = 10. The pool is the counterparty of every trade, so
        # the buyer side trades exactly Q_t.
        rows = [
            self._row("area_pv", "seller", 10.0, externality=10.0, p_cf=15.0),
            self._row("area_h1", "buyer", 4.0),
            self._row("area_h2", "buyer", 6.0),
        ]
        with caplog.at_level(logging.WARNING,
                             logger="amm-execution-node.penalties"):
            result = redistribution(rows, clearing_price=16.0,
                                    traded_quantity_kwh=10.0)

        assert result["harmed_side_kwh"] == pytest.approx(10.0)
        assert result["harmed_side_kwh"] == pytest.approx(
            result["compensated_ct"] / 1.0)   # damage is 1.0 ct/kWh here
        assert caplog.records == []

    def test_a_harmed_side_that_misses_the_traded_quantity_is_reported(
            self, caplog):
        """What the field is for.

        If the harmed set were ever built from both market sides again — the
        error the original specification contained — this sum would come out
        at roughly 2*Q_t. `budget_balance_ct` cannot reveal that: the pool is
        distributed in full either way and still reads 0.00. This is the only
        place the mistake becomes visible.

        Here the mismatch is forced from the other direction, by passing a
        `traded_quantity_kwh` the rows do not support — the harmed side holds
        10.0 kWh against a claimed Q_t of 20.0.
        """
        rows = [
            self._row("area_pv", "seller", 10.0, externality=10.0, p_cf=15.0),
            self._row("area_h1", "buyer", 4.0),
            self._row("area_h2", "buyer", 6.0),
        ]
        with caplog.at_level(logging.WARNING,
                             logger="amm-execution-node.penalties"):
            result = redistribution(rows, clearing_price=16.0,
                                    traded_quantity_kwh=20.0)

        # the true sum, not the claimed one
        assert result["harmed_side_kwh"] == pytest.approx(10.0)
        assert len(caplog.records) == 1
        assert "harmed set may span both market sides" in \
            caplog.records[0].getMessage()
        # reported, never raised: a violated invariant must not become a 500
        assert result["budget_balance_ct"] == pytest.approx(0.0, abs=1e-6)

    def test_a_round_without_deviators_reports_no_harmed_side_volume(self):
        # No deviator means no harmed side, so there is no volume to sum.
        rows = [self._row("area_pv", "seller", 10.0),
                self._row("area_h1", "buyer", 10.0)]
        result = redistribution(rows, clearing_price=16.0,
                                traded_quantity_kwh=10.0)

        assert result["harmed_side"] is None
        assert result["harmed_side_kwh"] == 0.0

