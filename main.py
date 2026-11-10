import numpy as np
import pandas as pd

# 1. Simulation Parameters
N = 10000
fixed_costs = 2000000
var_cost_per_bu = 0.20
F_0 = 4.00
cash_reserves = 500000  # Max margin buffer before forced liquidation

prob_drought = 0.15
prob_price_spike = 0.05  # 5% chance of a transient "tweet shock" margin spike
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
spike_state = rng.binomial(1, prob_price_spike, N)  # Trigger for margin spike

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

# Simulate pre-harvest peak price for margin calls
# If spike occurs, prices surge between $8 and $15. Otherwise, peak is slightly above F_T or F_0.
F_peak = np.where(
    spike_state == 1,
    rng.uniform(8.00, 15.00, N),
    np.maximum(F_T, F_0 + rng.uniform(0, 0.50, N))
)

B_T = rng.normal(basis_mean, basis_std, N)
S_T = F_T + B_T

# Calculate Revenue and Costs (separated to accurately target sales-proceeds variance)
revenue_unhedged = Q_T * S_T
var_costs = Q_T * var_cost_per_bu
total_costs = fixed_costs + var_costs
profit_unhedged = revenue_unhedged - total_costs

# 3. Grid Searches for Optimal Hedge Ratios (CME vs. OTC)
print("Starting Grid Searches (Targeting Minimum Sales-Proceeds Variance)...\n")
test_ratios = np.arange(0.0, 1.05, 0.05)

# A. Search CME (Vulnerable to Liquidation)
best_cme_std = float('inf')
best_cme_short = 0
best_cme_put = 0
best_cme_proceeds = None
best_cme_profit = None

for short_pct in test_ratios:
    for put_pct in test_ratios:
        if short_pct + put_pct > 1.20:
            continue

        short_vol = expected_yield * short_pct
        put_vol = expected_yield * put_pct

        # CME Liquidation Logic
        margin_call = short_vol * np.maximum(F_peak - F_0, 0)
        liquidated = margin_call > cash_reserves

        # If liquidated, farmer takes a realized loss at F_peak and loses the hedge for harvest
        cme_payoff = np.where(
            liquidated,
            short_vol * (F_0 - F_peak),
            short_vol * (F_0 - F_T)
        )
        put_payoff = put_vol * np.maximum(put_strike - F_T, 0) - (put_vol * put_premium)

        proceeds_sim = revenue_unhedged + cme_payoff + put_payoff
        current_std = np.std(proceeds_sim)  # Target: Sales-Proceeds Variance

        if current_std < best_cme_std:
            best_cme_std = current_std
            best_cme_short = short_pct
            best_cme_put = put_pct
            best_cme_proceeds = proceeds_sim
            best_cme_profit = proceeds_sim - total_costs

# B. Search OTC (Protected by Credit Line)
best_otc_std = float('inf')
best_otc_short = 0
best_otc_put = 0
best_otc_proceeds = None
best_otc_profit = None

for otc_pct in test_ratios:
    for put_pct in test_ratios:
        if otc_pct + put_pct > 1.20:
            continue

        otc_vol = expected_yield * otc_pct
        put_vol = expected_yield * put_pct

        # OTC survives margin spikes via the embedded credit line fee
        otc_payoff = otc_vol * (F_0 - F_T) - (otc_vol * otc_fee_per_bu)
        put_payoff = put_vol * np.maximum(put_strike - F_T, 0) - (put_vol * put_premium)

        proceeds_sim = revenue_unhedged + otc_payoff + put_payoff
        current_std = np.std(proceeds_sim)  # Target: Sales-Proceeds Variance

        if current_std < best_otc_std:
            best_otc_std = current_std
            best_otc_short = otc_pct
            best_otc_put = put_pct
            best_otc_proceeds = proceeds_sim
            best_otc_profit = proceeds_sim - total_costs

# 4. Final Strategy Comparison
# Calculate 100% CME Baseline with forced liquidation mechanics
cme_100_vol = expected_yield * 1.0
cme_100_margin = cme_100_vol * np.maximum(F_peak - F_0, 0)
cme_100_liquidated = cme_100_margin > cash_reserves
cme_100_payoff = np.where(cme_100_liquidated, cme_100_vol * (F_0 - F_peak), cme_100_vol * (F_0 - F_T))
proceeds_100_cme = revenue_unhedged + cme_100_payoff
profit_100_cme = proceeds_100_cme - total_costs

data = [
    {
        "Strategy": "Unhedged Baseline",
        "Proceeds Std Dev (Target)": float(np.std(revenue_unhedged)),
        "Expected Profit": float(np.mean(profit_unhedged)),
        "5th Percentile Profit": float(np.percentile(profit_unhedged, 5))
    },
    {
        "Strategy": "100% CME Futures",
        "Proceeds Std Dev (Target)": float(np.std(proceeds_100_cme)),
        "Expected Profit": float(np.mean(profit_100_cme)),
        "5th Percentile Profit": float(np.percentile(profit_100_cme, 5))
    },
    {
        "Strategy": f"Best CME Feasible ({best_cme_short:.0%} CME + {best_cme_put:.0%} Puts)",
        "Proceeds Std Dev (Target)": float(best_cme_std),
        "Expected Profit": float(np.mean(best_cme_profit)),
        "5th Percentile Profit": float(np.percentile(best_cme_profit, 5))
    },
    {
        "Strategy": f"Best OTC Feasible ({best_otc_short:.0%} OTC + {best_otc_put:.0%} Puts)",
        "Proceeds Std Dev (Target)": float(best_otc_std),
        "Expected Profit": float(np.mean(best_otc_profit)),
        "5th Percentile Profit": float(np.percentile(best_otc_profit, 5))
    }
]

results_df = pd.DataFrame(data)

for col in results_df.columns[1:]:
    results_df[col] = results_df[col].apply(lambda x: f"${x:,.0f}")

print("--- FINAL STRATEGY COMPARISON ---")
print(results_df.to_string(index=False))