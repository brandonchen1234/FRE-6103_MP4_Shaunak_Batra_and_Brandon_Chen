import numpy as np
import pandas as pd

# 1. Simulation Parameters
N = 10000
fixed_costs = 2000000
var_cost_per_bu = 0.20
F_0 = 4.00

prob_drought = 0.15
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

B_T = rng.normal(basis_mean, basis_std, N)
S_T = F_T + B_T

revenue_unhedged = Q_T * S_T
costs = fixed_costs + (Q_T * var_cost_per_bu)
profit_unhedged = revenue_unhedged - costs

# 3. Grid Search for Optimal Hedge Ratios
print("Starting Grid Search for Minimum Variance...\n")
best_std = float('inf')
best_otc = 0
best_put = 0
best_profit_array = None

grid_results = []

test_ratios = np.arange(0.0, 1.05, 0.05)

for otc in test_ratios:
    for put in test_ratios:
        # Cap total volume hedged to prevent absurd over-hedging
        if otc + put > 1.20:
            continue

        otc_vol = expected_yield * otc
        put_vol = expected_yield * put

        otc_payoff = otc_vol * (F_0 - F_T) - (otc_vol * otc_fee_per_bu)
        put_payoff = put_vol * np.maximum(put_strike - F_T, 0) - (put_vol * put_premium)

        profit_sim = profit_unhedged + otc_payoff + put_payoff
        current_std = np.std(profit_sim)
        current_mean = np.mean(profit_sim)

        grid_results.append({
            "OTC %": f"{otc:.0%}",
            "Put %": f"{put:.0%}",
            "Std Dev (Risk)": current_std,
            "Expected Profit": current_mean
        })

        if current_std < best_std:
            best_std = current_std
            best_otc = otc
            best_put = put
            best_profit_array = profit_sim

# 4. Display Top 5 Grid Search Results
grid_df = pd.DataFrame(grid_results)
grid_df = grid_df.sort_values(by="Std Dev (Risk)").head(5)

for col in ["Std Dev (Risk)", "Expected Profit"]:
    grid_df[col] = grid_df[col].apply(lambda x: f"${x:,.0f}")

print("--- Top 5 Grid Search Results (Lowest Variance) ---")
print(grid_df.to_string(index=False))
print(f"\n>> OPTIMAL MINIMUM VARIANCE FOUND: {best_otc:.0%} OTC Futures + {best_put:.0%} Puts <<\n")

# 5. Final Strategy Comparison
futures_payoff_100 = expected_yield * 1.0 * (F_0 - F_T)
profit_100_futures = profit_unhedged + futures_payoff_100

data = [
    {
        "Strategy": "Unhedged Baseline",
        "Expected Profit": float(np.mean(profit_unhedged)),
        "Std Dev (Risk)": float(np.std(profit_unhedged)),
        "5th Percentile (Worst 5%)": float(np.percentile(profit_unhedged, 5))
    },
    {
        "Strategy": "100% CME Futures",
        "Expected Profit": float(np.mean(profit_100_futures)),
        "Std Dev (Risk)": float(np.std(profit_100_futures)),
        "5th Percentile (Worst 5%)": float(np.percentile(profit_100_futures, 5))
    },
    {
        "Strategy": f"Grid Search Optimal ({best_otc:.0%} OTC + {best_put:.0%} Puts)",
        "Expected Profit": float(np.mean(best_profit_array)),
        "Std Dev (Risk)": float(np.std(best_profit_array)),
        "5th Percentile (Worst 5%)": float(np.percentile(best_profit_array, 5))
    }
]

results_df = pd.DataFrame(data)

for col in results_df.columns[1:]:
    results_df[col] = results_df[col].apply(lambda x: f"${x:,.0f}")

print("--- FINAL STRATEGY COMPARISON ---")
print(results_df.to_string(index=False))