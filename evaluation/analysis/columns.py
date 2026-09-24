"""One line per column of every table `dr_analysis.py` writes.

The writer refuses a table with a column that is not documented here, so
`columns.md` cannot fall behind the CSVs.
"""

#: Readings the tables depend on, stated once at the top of columns.md.
READINGS = (
    "Equation numbers are those of `Chapters 1-4 2026-09-24 v2`: (4.3) "
    "shortfall, (4.4)/(4.5) seller/buyer externality, (4.6)/(4.7) the "
    "long-side conditions of Proposition 1, (4.8) the levy IR condition.",
    "Energy-based model throughout: Q = min(E, D_hat) and the pro-rata "
    "allocation do not depend on the price, so a band sensitivity reprices "
    "the same book.",
    "epsilon_t is the elasticity of the side's own margin in the ratio: "
    "|f'(x)| x / (p - K_lower) for sellers (Section 4.3.3, 2.036 at x = "
    "1.25), |f'(x)| x / (K_upper - p) for buyers. The buyer form is this "
    "analysis's reading; it is the one under which the long-side condition "
    "has the same shape on both sides.",
    "The eight cases are side x side role x direction. The seller is the "
    "short side in a SUPPLY_LIMITED round, the buyer in a DEMAND_LIMITED "
    "one. BALANCED slots have no short side and are left out of DR2.",
    "Under-delivering sellers (`sellers_underdeliver`, the b2_short_* "
    "cells) are read as row 2 of Table 4.2: an over-report on the short "
    "side whose truthful report is `actual_kwh`. Only supply-limited slots "
    "are replayed for them, as for the withholding sellers.",
    "Forgone margin is defined as price effect minus gross gain, so the two "
    "always sum to the gross gain. Without a regime switch it equals "
    "(p_cf - K_lower) * W for a seller and (K_upper - p_cf) * U for a buyer.",
    "The joint counterfactual of DR4 puts back the regime's penalised side "
    "only: sellers in a SUPPLY_LIMITED round, buyers in a DEMAND_LIMITED "
    "one, which is where the node charges an externality at all.",
    "Nothing here reads configured parameters where the run recorded "
    "applied ones: gamma, eta_relative and K_sho come from the slot CSV, the "
    "band from `sigmoid_effective`. Runs that predate those records fall "
    "back to the manifest -- `config.gamma` / `config.eta_relative` only "
    "when both are set explicitly, `config.sigmoid` for the band -- and "
    "`census.param_source` / `census.sigmoid_source` say which.",
)

COLUMNS = {
    "census": (
        ("cell", "campaign cell, from the manifest"),
        ("seed", "replicate seed"),
        ("run_id", "run id, from the manifest"),
        ("path", "run directory the CSVs were read from"),
        ("repo_sha", "SHA the run was produced at"),
        ("n_manifest_entries", "entries in the run's manifest; > 1 means a "
                               "re-run was appended and the last one is used"),
        ("sigmoid_source", "sigmoid_effective, or config.sigmoid for runs "
                           "before 17.09."),
        ("k_upper", "K_upper of the band the run was priced at"),
        ("k_lower", "K_lower"),
        ("theta", "theta"),
        ("steepness", "B"),
        ("executed", "whether the run executed (block 2 and the sweep)"),
        ("param_source", "where gamma / eta / K_sho come from: slot_csv "
                         "(gamma_eff etc.), or manifest_config for slot CSVs "
                         "written before those columns (config.gamma and "
                         "config.eta_relative set explicitly); empty for "
                         "runs that did not execute"),
        ("arm", "manifest deviation arm, empty for block 1"),
        ("n_slots", "rows in the slot CSV"),
        ("n_cleared", "slots with a clearing_price_ct"),
        ("n_supply_limited", "cleared slots with E < D_hat (determine_round_"
                             "type on the recorded aggregates)"),
        ("n_demand_limited", "cleared slots with E > D_hat"),
        ("n_balanced", "cleared slots with E = D_hat within 1e-9"),
        ("n_no_trade", "slots that did not clear"),
        ("manifest_census_agrees", "the three counts equal the manifest's "
                                   "checks.round_type_census"),
        ("exec_round_type_agrees", "for executed runs, round_type_exec equals "
                                   "the recomputed round type in every "
                                   "executed slot"),
        ("gamma_eff", "gamma the execution node applied (distinct values)"),
        ("eta_relative_eff", "relative deadband applied (distinct values)"),
        ("k_sho_ct_per_kwh", "K_sho = gamma * K_upper applied (distinct "
                             "values)"),
    ),
    "dr2_cases": (
        ("case", "side_role_direction, one of the eight rows of Table 4.2"),
        ("table_4_2_row", "row of Table 4.2: 1 seller short withhold, 2 "
                          "seller short over-report, 3 seller long withhold, "
                          "4 seller long over-report, 5 buyer short "
                          "under-report, 6 buyer short over-report, 7 buyer "
                          "long under-report, 8 buyer long over-report"),
        ("side", "seller or buyer"),
        ("side_role", "short or long side of the slot"),
        ("regime", "round type at the truthful point"),
        ("direction", "underreport (withhold / under-report) or overreport"),
        ("type", "all, household or battery (area id prefix battery-)"),
        ("n", "participant-slots of b2_noise_off (five seeds) in the case"),
        ("share_net_max_pos", "share with a net gain > 1e-6 ct somewhere on "
                              "the grid"),
        ("net_max_ct_median", "median of the largest net gain over the grid, "
                              "ct"),
        ("net_max_ct_p90", "90th percentile of the same, ct"),
        ("net_max_ct_max", "maximum of the same, ct"),
        ("net_max_rel_median", "median of net_max / truthful payoff"),
        ("net_max_rel_p90", "90th percentile of net_max / truthful payoff"),
        ("net_max_rel_max", "maximum of net_max / truthful payoff"),
        ("delta_star_rel_median", "median maximiser delta* / claim"),
        ("delta_active_rel_median", "median of delta_active / claim: the "
                                    "smallest deviation at which any "
                                    "unrounded penalty is non-zero (bisected "
                                    "between grid points); inf if none on "
                                    "the grid"),
        ("n_delta_active_inf", "participant-slots with no active penalty on "
                               "the grid"),
        ("delta_zero_rel_median", "median of delta_zero / claim: the smallest "
                                  "deviation above the maximiser with net "
                                  "gain <= 0 (bisected); 0 where no "
                                  "deviation pays, inf where it pays over "
                                  "the whole grid"),
        ("n_delta_zero_inf", "participant-slots whose net gain stays "
                             "positive over the grid"),
        ("net_at_active_ct_median", "median net gain at delta_active, ct"),
        ("share_switch_at_max", "share whose maximiser lies in a switched "
                                "regime (the propositions do not cover it)"),
        ("share_switch_any", "share whose sweep switches the regime anywhere "
                             "on the grid"),
        ("prop2_n_violations", "short-side sellers only: participant-slots "
                               "with a net gain beyond delta_active, within "
                               "the regime, above net(delta_active) + 1e-6 "
                               "ct; Proposition 2 predicts 0"),
    ),
    "dr2_curves": (
        ("case", "as in dr2_cases"),
        ("delta_rel", "delta / claim, grid point"),
        ("n", "participant-slots in the case"),
        ("gross_ct_median", "median gross gain (utility without penalties "
                            "minus truthful utility), ct"),
        ("gross_ct_p90", "90th percentile gross gain, ct"),
        ("penalty_ct_median", "median penalty (shortfall + externality), ct"),
        ("penalty_ct_p90", "90th percentile penalty, ct"),
        ("net_ct_median", "median net gain, ct"),
        ("net_ct_p90", "90th percentile net gain, ct"),
        ("gross_rel_median", "median gross gain / truthful payoff"),
        ("gross_rel_p90", "90th percentile gross gain / truthful payoff"),
        ("penalty_rel_median", "median penalty / truthful payoff"),
        ("penalty_rel_p90", "90th percentile penalty / truthful payoff"),
        ("net_rel_median", "median net gain / truthful payoff"),
        ("net_rel_p90", "90th percentile net gain / truthful payoff"),
        ("share_regime_switch", "share of the case whose regime differs from "
                                "the truthful one at this grid point"),
    ),
    "dr2_conditions": (
        ("side", "seller or buyer"),
        ("side_role", "long (Proposition 1, Eqs. 4.6/4.7) or short "
                      "(Section 4.3.4, sellers)"),
        ("regime", "round type"),
        ("type", "all, household or battery"),
        ("condition", "eps > (1-sigma)/sigma on the long side, eps*sigma > 1 "
                      "for short-side sellers; sigma = own report / own "
                      "side's total"),
        ("n", "participant-slots of b2_noise_off"),
        ("n_holds", "participant-slots where the condition holds: "
                    "withholding / under-reporting pays at the margin"),
        ("n_fails", "participant-slots where it fails; on the long side this "
                    "is where over-reporting pays at the margin"),
        ("sigma_max", "largest sigma in the group"),
        ("eps_median", "median epsilon_t"),
        ("eps_max", "largest epsilon_t"),
        ("lhs_max", "largest left-hand side (epsilon_t, or epsilon_t * "
                    "sigma)"),
        ("closest_margin", "smallest |lhs - rhs| in the group"),
        ("closest_margin_signed", "lhs - rhs at that participant-slot"),
        ("closest_sigma", "sigma at that participant-slot"),
    ),
    "dr2_band_sensitivity": (
        ("steepness", "B of the repriced band"),
        ("theta", "theta of the repriced band; K bounds unchanged"),
        ("kind", "condition (Proposition 1 / Section 4.3.4 count) or case "
                 "(share with net_max > 0)"),
        ("name", "the condition or the case"),
        ("n", "participant-slots"),
        ("n_positive", "condition holds / net_max > 1e-6 ct"),
        ("share", "n_positive / n"),
    ),
    "dr2_named": (
        ("cell", "block-2 or sweep cell with a named deviator"),
        ("seed", "replicate seed"),
        ("run_id", "run id"),
        ("arm", "sellers_withhold, sellers_underdeliver or "
                "buyers_underreport"),
        ("share", "configured deviation share"),
        ("k", "configured coalition size"),
        ("gamma", "configured gamma"),
        ("eta_relative", "configured relative deadband"),
        ("deviator", "area id of the named deviator"),
        ("n_active_slots", "slots in which the deviator trades on the arm's "
                           "side"),
        ("n_other_side_slots", "slots in which it trades on the other side "
                               "(accidental layer only, not replayed)"),
        ("n_replayed", "arm-side slots replayed against the truthful "
                       "counterfactual, regime-switching ones included"),
        ("n_not_replayed", "arm-side demand-limited slots of a seller arm: "
                           "counted, not replayed"),
        ("n_regime_switch_excluded", "replayed slots whose counterfactual "
                                     "regime differs from the recorded one; "
                                     "left out of every sum below except "
                                     "penalty_all_ct and "
                                     "penalty_other_side_ct"),
        ("gross_gain_ct", "sum over the replayed in-regime slots of gross "
                          "utility (no penalties) at the recorded report "
                          "minus at the truthful one, ct"),
        ("price_effect_ct", "sum of (p - p_cf) * s (sellers) or (p_cf - p) * "
                            "q (buyers), ct"),
        ("forgone_margin_ct", "price_effect_ct - gross_gain_ct: the margin "
                              "(saving, for buyers) on the energy the "
                              "deviation moved, ct"),
        ("penalty_all_ct", "recorded total penalty in all arm-side slots, "
                           "ct"),
        ("penalty_in_regime_ct", "recorded total penalty in the replayed "
                                 "in-regime slots only, ct"),
        ("penalty_other_side_ct", "recorded penalty in the other-side slots, "
                                  "ct"),
        ("max_price_replay_error", "largest |replayed - recorded| price at "
                                   "the recorded report, ct"),
        ("net_gain_ct", "gross_gain_ct - penalty_in_regime_ct"),
        ("slots_with_absent_deviator", "manifest deviation."
                                       "slots_with_absent_deviator"),
    ),
    "dr3_ir": (
        ("cell", "campaign cell, all seeds"),
        ("side", "seller or buyer"),
        ("group", "named (manifest deviation.deviators) or other"),
        ("cause", "any, or the adjustment whose removal alone restores u >= "
                  "0: levy (grey seller settled below p), shortfall, "
                  "externality, unneeded_energy (buyer with q > actual); "
                  "combined where no single removal does"),
        ("n", "participant-slots of the cell, side and group"),
        ("n_violations", "participant-slots with u < -1e-9 attributed to the "
                         "cause (one violation can have several causes)"),
        ("share", "n_violations / n"),
        ("min_u_ct", "smallest u among those violations, ct"),
        ("sum_negative_u_ct", "sum of u over those violations, ct"),
    ),
    "dr3_settlement": (
        ("cell", "campaign cell, all seeds"),
        ("side", "seller or buyer"),
        ("n", "participant-slots with a positive quantity"),
        ("rate_min_ct", "minimum settlement rate: seller (r s - Phi - phi + "
                        "c) / s, buyer (r_b q + phi - c) / q, ct/kWh"),
        ("rate_p10_ct", "10th percentile"),
        ("rate_median_ct", "median"),
        ("rate_mean_ct", "mean"),
        ("rate_p90_ct", "90th percentile"),
        ("rate_max_ct", "maximum"),
    ),
    "dr3_eq48": (
        ("part", "measured (runs with multipliers on) or exposure (reference "
                 "cells, independent of the configured levy)"),
        ("cell", "campaign cell, all seeds"),
        ("levy", "exposure: the levy lambda the threshold is read at"),
        ("n", "measured: grey seller rows; exposure: cleared slots with grey "
              "supply"),
        ("n_below", "measured: rows with (1 - lambda_t) p < K_lower; "
                    "exposure: slots with p < K_lower / (1 - lambda)"),
        ("share_below", "n_below / n"),
        ("lambda_min", "measured: smallest lambda_t = 1 - grey_final / p"),
        ("lambda_max", "measured: largest lambda_t"),
        ("min_price_ct", "exposure: lowest clearing price among the slots"),
        ("min_grey_final_ct", "measured: lowest grey seller rate"),
        ("price_threshold_ct", "exposure: K_lower / (1 - lambda)"),
        ("ratio_threshold", "exposure: the ratio above which p falls below "
                            "that threshold, for the run's band"),
        ("n_slots_below", "slots with at least one row / the slot below the "
                          "threshold"),
    ),
    "dr4_layers": (
        ("layer", "externality, shortfall or energy_origin"),
        ("cell", "campaign cell, all seeds"),
        ("mode", "energy_origin: multiplier formulation"),
        ("sides", "energy_origin: seller or both"),
        ("n_slots", "executed slots (energy_origin: cleared slots with "
                    "multipliers on)"),
        ("n_slots_nonzero", "slots with a pool / a retained shortfall / a "
                            "surplus above 1e-6 ct"),
        ("n_imbalanced", "externality: slots with |pool - compensated| > "
                         "1e-6 ct"),
        ("n_undistributed", "externality: slots with a pool and nothing "
                            "compensated (total damage <= eps)"),
        ("n_budget_balance_nonzero", "externality: slots whose recorded "
                                     "budget_balance_ct is not 0"),
        ("sum_pool_ct", "sum of the pool (externality), of the retained "
                        "shortfall penalties, or of pool_surplus_ct"),
        ("sum_compensated_ct", "externality: sum of compensated_ct"),
        ("sum_balance_ct", "externality: sum of pool - compensated"),
        ("max_abs_balance_ct", "externality: largest |pool - compensated|"),
        ("min_ct", "shortfall: smallest retained sum per slot (>= 0 by "
                   "construction); energy_origin: smallest surplus"),
        ("max_ct", "energy_origin: largest surplus"),
    ),
    "dr4_coverage": (
        ("cell", "executed cell, all seeds"),
        ("arm", "manifest deviation arm"),
        ("n_deviators", "bucket of the node's externality deviators in the "
                        "slot: 0 (deadband-only withholding, no pool), 1, 2, "
                        "3-5, 6-10, >10"),
        ("deviator_class", "all_named, all_accidental, mixed, or none"),
        ("n_slots", "slots in the group"),
        ("c_pen_median", "median C_pen = compensated / damage against the "
                         "joint penalizable counterfactual"),
        ("c_pen_min", "minimum C_pen"),
        ("c_pen_max", "maximum C_pen"),
        ("c_tot_median", "median C_tot = compensated / damage against the "
                         "joint total counterfactual (no deadband)"),
        ("c_tot_min", "minimum C_tot"),
        ("c_tot_max", "maximum C_tot"),
        ("sum_pool_ct", "sum of penalty_pool_ct"),
        ("sum_compensated_ct", "sum of compensated_ct"),
        ("sum_damage_pen_ct", "sum of |p - p_cf,joint| Q, penalizable"),
        ("sum_damage_tot_ct", "sum of |p - p_cf,joint| Q, total"),
    ),
    "dr5_fairness": (
        ("cell", "block-1 cell, all seeds"),
        ("side", "seller or buyer"),
        ("side_role", "short (filled completely), long, or balanced"),
        ("trivial", "True on the short side, where J = 1 holds trivially"),
        ("n_slots", "slots with this side in this role"),
        ("n_classes", "classes of equals (same side, preference status and, "
                      "for served_first, partner capacity share c)"),
        ("n_singleton", "classes with one member"),
        ("n_members_nonsingleton", "members of classes with two or more"),
        ("j_class_min", "smallest Jain index (Eq. 2.5) of fill rates within "
                        "a non-singleton class; predicted exactly 1"),
        ("j_class_median", "median of the same"),
        ("residual_n", "long side: served_first members checked against "
                       "fill = c + (1 - c) r"),
        ("residual_skipped", "served_first members in slots with no "
                             "non-pair member on their side (r undefined)"),
        ("residual_max_abs_dev", "largest |fill - (c + (1 - c) r)|; fill_rate "
                                 "and requested_kwh carry six decimals"),
        ("j_all_median", "median Jain index across all participants of the "
                         "side, per slot (descriptive)"),
        ("j_all_min", "smallest of the same"),
    ),
    "dr5_preferences": (
        ("cell", "block-1 cell, all seeds"),
        ("n_named", "participant-slots of named participants"),
        ("n_served_first", "of them served first"),
        ("share_served_first", "n_served_first / n_named"),
        ("claims_kwh", "sum of the named participants' requested_kwh"),
        ("served_kwh", "sum of the pair quantity min(own, partner) over the "
                       "served-first named participants (each pair counts "
                       "for both members)"),
        ("share_energy_served_first", "served_kwh / claims_kwh"),
        ("n_order_pro_rata_first", "not served first: the order is "
                                   "pro_rata_first (or preferences off)"),
        ("n_not_reciprocated", "not served first: the partner nominates "
                               "someone else (read from any row of the "
                               "partner in the run)"),
        ("n_partner_not_trading", "not served first: the partner has no row "
                                  "in the slot"),
        ("n_partner_same_side", "not served first: the partner trades on the "
                                "same side"),
        ("n_partner_never_in_run", "not served first: the partner has no row "
                                   "anywhere in the run"),
        ("n_other", "not served first for none of the reasons above "
                    "(expected 0)"),
    ),
    "coalitions": (
        ("arm", "sellers_withhold or buyers_underreport"),
        ("cell", "b2_*_s25 (k = 1), b2_*_k02, b2_*_k05, b2_*_k10"),
        ("k", "coalition size"),
        ("seed", "replicate seed, or all"),
        ("deviators", "the coalition, from the manifest"),
        ("coalition_penalty_ct", "sum of the members' externality penalties "
                                 "on the arm's side, ct"),
        ("penalized_kwh", "sum of the penalised quantity W (sellers, after "
                          "the deadband) or U (buyers) over the same rows"),
        ("penalty_per_kwh_ct", "coalition_penalty_ct / penalized_kwh"),
        ("k1_member", "the k = 1 deviator, a member of every coalition"),
        ("k1_penalty_ct", "its penalty within this cell, ct"),
        ("k1_penalized_kwh", "its penalised quantity within this cell"),
        ("k1_penalty_per_kwh_ct", "k1_penalty_ct / k1_penalized_kwh"),
    ),
}


def names(table: str) -> list:
    return [name for name, _doc in COLUMNS[table]]


def markdown(notes=()) -> str:
    lines = ["# Columns of the DR2-DR5 analysis", "",
             "Written by `evaluation/analysis/dr_analysis.py`. One line per "
             "column; `checks.json` carries every check with its maximum "
             "deviation, `manifest.json` the inputs.", "", "## Readings", ""]
    lines += [f"- {text}" for text in READINGS]
    if notes:
        lines += ["", "## Notes from this run", ""]
        lines += [f"- {text}" for text in notes]
    for table, columns in COLUMNS.items():
        lines += ["", f"## {table}.csv", ""]
        lines += [f"- `{name}`: {doc}" for name, doc in columns]
    return "\n".join(lines) + "\n"
