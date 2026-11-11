import numpy as np
import pandas as pd

# 1. Simulation Parameters
N = 10000
fixed_costs = 2000000
var_cost_per_bu = 0.20
F_0 = 4.00
cash_reserves = 500000  # Available for variation margin after initial margin and put premiums
otc_credit_limit = 550000  # Maximum variation margin covered by the OTC credit line

prob_drought = 0.15
prob_price_spike = 0.05
yield_normal_mean, yield_normal_std = 1080000, 120000
yield_drought_mean, yield_drought_std = 550000, 150000

F_T_normal_mean, F_T_normal_std = 3.78, 3.78 * 0.30 * np.sqrt(0.5)
F_T_drought_mean, F_T_drought_std = 5.25, 5.25 * 0.30 * np.sqrt(0.5)
basis_mean, basis_std = -0.15, 0.20

expected_yield = 1000000
put_strike = 4.00
put_premium = 0.33
otc_fee_per_bu = 0.03

rng = np.random.default_rng(seed=42)

# 2. Simulate Base Variables
drought_state = rng.binomial(1, prob_drought, N)
spike_state = rng.binomial(1, prob_price_spike, N)

Q_T = np.where(
    drought_state == 1,
    rng.normal(yield_drought_mean, yield_drought_std, N),
    rng.normal(yield_normal_mean, yield_normal_std, N)
)
Q_T = np.maximum(Q_T, 0)

F_T = np.where(
    drought_state == 1,
    rng.normal(F_T_drought_mean, F_T_drought_std, N),
    rng.normal(F_T_normal_mean, F_T_normal_std, N)
)
F_T = np.maximum(F_T, 0)

# Simulate the peak futures price over the hedge period for margin calls
F_peak = np.where(
    spike_state == 1,
    rng.uniform(8.00, 15.00, N),
    F_0 + rng.uniform(0, 0.50, N)
)
F_peak = np.maximum(F_peak, F_T)
peak_margin_per_bu = np.maximum(F_peak - F_0, 0)

B_T = rng.normal(basis_mean, basis_std, N)
S_T = F_T + B_T

# Calculate sales revenue and production costs separately
revenue_unhedged = Q_T * S_T
var_costs = Q_T * var_cost_per_bu
total_costs = fixed_costs + var_costs
profit_unhedged = revenue_unhedged - total_costs

# 3. Grid Searches for Optimal Hedge Ratios (CME vs. OTC)
print("Starting Grid Searches (Targeting Minimum Sales-Proceeds Variance)...\n")

# Each step is 5% of expected yield. A combined 24 steps equals 120%.
test_steps = range(21)
max_combined_steps = 24

# A. Search CME (Vulnerable to Liquidation)
best_cme_std = float("inf")
best_cme_short = 0
best_cme_put = 0
best_cme_profit = None

for short_step in test_steps:
    for put_step in test_steps:
        if short_step + put_step > max_combined_steps:
            continue

        short_pct = short_step / 20
        put_pct = put_step / 20
        short_vol = expected_yield * short_pct
        put_vol = expected_yield * put_pct

        margin_call = short_vol * peak_margin_per_bu
        liquidated = margin_call > cash_reserves

        # Assume the broker closes the futures position at the simulated peak
        # if the margin buffer is breached. Any puts remain in place.
        cme_payoff = np.where(
            liquidated,
            short_vol * (F_0 - F_peak),
            short_vol * (F_0 - F_T)
        )
        put_payoff = put_vol * (np.maximum(put_strike - F_T, 0) - put_premium)

        proceeds_sim = revenue_unhedged + cme_payoff + put_payoff
        current_std = np.std(proceeds_sim)

        if current_std < best_cme_std:
            best_cme_std = current_std
            best_cme_short = short_pct
            best_cme_put = put_pct
            best_cme_profit = proceeds_sim - total_costs

# B. Search OTC (Subject to the Credit Line Limit)
best_otc_std = float("inf")
best_otc_short = 0
best_otc_put = 0
best_otc_profit = None

for otc_step in test_steps:
    for put_step in test_steps:
        if otc_step + put_step > max_combined_steps:
            continue

        otc_pct = otc_step / 20
        put_pct = put_step / 20
        otc_vol = expected_yield * otc_pct
        put_vol = expected_yield * put_pct

        # Only consider OTC positions the credit line can fund in every simulated scenario
        otc_margin = otc_vol * peak_margin_per_bu
        if np.any(otc_margin > otc_credit_limit):
            continue

        otc_payoff = otc_vol * (F_0 - F_T - otc_fee_per_bu)
        put_payoff = put_vol * (np.maximum(put_strike - F_T, 0) - put_premium)

        proceeds_sim = revenue_unhedged + otc_payoff + put_payoff
        current_std = np.std(proceeds_sim)

        if current_std < best_otc_std:
            best_otc_std = current_std
            best_otc_short = otc_pct
            best_otc_put = put_pct
            best_otc_profit = proceeds_sim - total_costs

# 4. Final Strategy Comparison
# Apply the same liquidation rule to the 100% CME hedge from class
cme_100_vol = expected_yield
cme_100_margin = cme_100_vol * peak_margin_per_bu
cme_100_liquidated = cme_100_margin > cash_reserves

cme_100_payoff = np.where(
    cme_100_liquidated,
    cme_100_vol * (F_0 - F_peak),
    cme_100_vol * (F_0 - F_T)
)
proceeds_100_cme = revenue_unhedged + cme_100_payoff
profit_100_cme = proceeds_100_cme - total_costs

# Include puts only to measure the effect of adding the OTC short
puts_only_payoff = expected_yield * (
    np.maximum(put_strike - F_T, 0) - put_premium
)
proceeds_puts_only = revenue_unhedged + puts_only_payoff
profit_puts_only = proceeds_puts_only - total_costs

data = [
    {
        "Strategy": "Unhedged Baseline",
        "Proceeds Std Dev (Target)": np.std(revenue_unhedged),
        "Expected Profit": np.mean(profit_unhedged),
        "5th Percentile Profit": np.percentile(profit_unhedged, 5)
    },
    {
        "Strategy": "100% CME Futures",
        "Proceeds Std Dev (Target)": np.std(proceeds_100_cme),
        "Expected Profit": np.mean(profit_100_cme),
        "5th Percentile Profit": np.percentile(profit_100_cme, 5)
    },
    {
        "Strategy": "100% Puts Only",
        "Proceeds Std Dev (Target)": np.std(proceeds_puts_only),
        "Expected Profit": np.mean(profit_puts_only),
        "5th Percentile Profit": np.percentile(profit_puts_only, 5)
    },
    {
        "Strategy": f"Best CME Mix ({best_cme_short:.0%} CME + {best_cme_put:.0%} Puts)",
        "Proceeds Std Dev (Target)": best_cme_std,
        "Expected Profit": np.mean(best_cme_profit),
        "5th Percentile Profit": np.percentile(best_cme_profit, 5)
    },
    {
        "Strategy": f"Best OTC Mix ({best_otc_short:.0%} OTC + {best_otc_put:.0%} Puts)",
        "Proceeds Std Dev (Target)": best_otc_std,
        "Expected Profit": np.mean(best_otc_profit),
        "5th Percentile Profit": np.percentile(best_otc_profit, 5)
    }
]

results_df = pd.DataFrame(data)

for col in results_df.columns[1:]:
    results_df[col] = results_df[col].apply(lambda x: f"${x:,.0f}")

print("--- FINAL STRATEGY COMPARISON ---")
print(results_df.to_string(index=False))
print(f"\n100% CME futures liquidation rate: {np.mean(cme_100_liquidated):.1%}")