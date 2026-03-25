"""
eroas_analysis.py

For a target campaign, compute eROAS on historical auction data, identify the
best bidding opportunities within a daily budget, and distribute them into
24 hourly buckets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")

# ── Config ─────────────────────────────────────────────────────────────────────

target_campaign_id = "0b76b55d-a017-4f77-a9a7-38fc41c90d2d"
daily_budget = 42500 * 2 # in cent
value_filtering_quantile = 0.95

#%%
# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_pickle(f"data/auction_history_cmp_{target_campaign_id}.pkl")
print(f"Loaded {len(df):,} auctions")

#%%
# ── 2. Compute per-auction metrics ─────────────────────────────────────────────

def compute_metrics(candidates: list, target_campaign_id: str) -> dict:
    """
    Returns a dict with keys: eROAS, impression_cost, best_quality_score.

    impression_cost logic
    ---------------------
    - Find the winner (auctionRank == 0).
    - If the winner IS the target campaign → impression_cost = pricingMetadata['nextAdScore']
      (what the target would actually pay under GSP, i.e. the next competitor's score).
    - Otherwise → impression_cost = winner's adScore
      (the score the target must beat to win this impression).

    eROAS logic
    -----------
    - Among all target-campaign entries, pick the one with the highest
      expected_purchase_value = itemPrice * adQualityScore.
    - eROAS = expected_purchase_value / impression_cost
    """
    # --- Target campaign entries ---
    target_entries = [c for c in candidates if c.get("campaignId") == target_campaign_id]
    if not target_entries:
        return {"eROAS": np.nan, "impression_cost": np.nan, "best_quality_score": np.nan}

    # Best expected purchase value across all target entries
    def epv(c):
        return (c.get("itemPrice") or 0.0) * (c.get("adQualityScore") or 0.0) * (c.get("predictedCtc") or 0.0)
    

    best_target = max(target_entries, key=epv)
    expected_purchase_value = epv(best_target)
    best_quality_score = max(c.get("adQualityScore") or 0.0 for c in target_entries)
    best_conversion_prob = max((c.get("adQualityScore") or 0.0) * (c.get("predictedCtc") or 0.0) for c in target_entries)

    # --- Winner entry ---
    winner_entries = [c for c in candidates if c.get("auctionRank") == 0]
    if not winner_entries:
        return {"eROAS": np.nan, "impression_cost": np.nan, "best_quality_score": best_quality_score}
    winner = winner_entries[0]

    # --- Impression cost ---
    if winner.get("campaignId") == target_campaign_id:
        pricing = winner.get("pricingMetadata") or {}
        impression_cost = pricing.get("nextAdScore", np.nan)
    else:
        impression_cost = winner.get("adScore", np.nan)

    # --- eROAS ---
    if impression_cost and impression_cost > 0:
        eroas = expected_purchase_value / impression_cost
    else:
        eroas = np.nan

    return {
        "eROAS": eroas,
        "impression_cost": impression_cost,
        "best_quality_score": best_quality_score,
        "best_conversion_prob": best_conversion_prob
    }


metrics = df["candidates"].apply(
    lambda cands: pd.Series(compute_metrics(cands, target_campaign_id))
)
df = pd.concat([df, metrics], axis=1)

print(f"eROAS computed: {df['eROAS'].notna().sum():,} valid / {len(df):,} total auctions")

#%%

q_levels = np.arange(0.0, 1.01, 0.05)
quantile_levels = np.arange(0.05, 1.01, 0.05)

eroas = df["eROAS"].dropna()
bqs = df["best_quality_score"].dropna()
bcp = df["best_conversion_prob"].dropna()

eroas_boundaries = eroas.quantile(q_levels).values
eroas_quantile_values = eroas.quantile(quantile_levels)
bqs_quantile_values = bqs.quantile(quantile_levels)
bcp_quantile_values = bcp.quantile(quantile_levels)

# ── Figure 1: quantile vs value for eROAS and best_quality_score ────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 5))

axes1[0].plot(quantile_levels, eroas_quantile_values, marker="o", markersize=4, color="steelblue", linewidth=2)
axes1[0].set_yscale("log")
axes1[0].set_xlabel("Quantile")
axes1[0].set_ylabel("eROAS (log scale)")
axes1[0].set_title("eROAS by Quantile (0.05 increments)")
axes1[0].set_xticks(quantile_levels)
axes1[0].set_xticklabels([f"{q:.2f}" for q in quantile_levels], rotation=45, ha="right", fontsize=7)
axes1[0].grid(True, linestyle="--", alpha=0.5)
for q, v in zip(quantile_levels, eroas_quantile_values):
    axes1[0].annotate(f"{v:.2f}", xy=(q, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)

axes1[1].plot(quantile_levels, bqs_quantile_values, marker="o", markersize=4, color="darkorange", linewidth=2)
axes1[1].set_yscale("log")
axes1[1].set_xlabel("Quantile")
axes1[1].set_ylabel("best_quality_score (log scale)")
axes1[1].set_title("best_quality_score by Quantile (0.05 increments)")
axes1[1].set_xticks(quantile_levels)
axes1[1].set_xticklabels([f"{q:.2f}" for q in quantile_levels], rotation=45, ha="right", fontsize=7)
axes1[1].grid(True, linestyle="--", alpha=0.5)
for q, v in zip(quantile_levels, bqs_quantile_values):
    axes1[1].annotate(f"{v:.4f}", xy=(q, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)

fig1.tight_layout()
fig1.show()

# ── Figure 2: avg best_quality_score per eROAS quantile bucket ──────────────
df_valid = df[df["eROAS"].notna() & df["best_quality_score"].notna()].copy()
df_valid["eroas_bucket"] = pd.cut(
    df_valid["eROAS"],
    bins=eroas_boundaries,
    labels=[f"{v:.2f}" for v in eroas_quantile_values],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("eroas_bucket", observed=True)["best_quality_score"].mean()

fig2, ax2 = plt.subplots(figsize=(14, 5))
bars = ax2.bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2.set_xticks(range(len(avg_bqs)))
ax2.set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2.set_xlabel("eROAS (quantile bucket upper bound)")
ax2.set_ylabel("Avg best_quality_score")
ax2.set_title("Avg best_quality_score per eROAS Quantile Bucket (0.05 increments)")
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)

fig2.tight_layout()
fig2.show()


# ── Figure 3: avg best_quality_score per eROAS quantile bucket ──────────────
df_valid = df[df["eROAS"].notna() & df["best_conversion_prob"].notna()].copy()
df_valid["eroas_bucket"] = pd.cut(
    df_valid["eROAS"],
    bins=eroas_boundaries,
    labels=[f"{v:.2f}" for v in eroas_quantile_values],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("eroas_bucket", observed=True)["best_conversion_prob"].mean()

fig2, ax2 = plt.subplots(figsize=(14, 5))
bars = ax2.bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2.set_xticks(range(len(avg_bqs)))
ax2.set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2.set_xlabel("eROAS (quantile bucket upper bound)")
ax2.set_ylabel("Avg best_conversion_prob")
ax2.set_title("Avg best_conversion_prob per eROAS Quantile Bucket (0.05 increments)")
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)

fig2.tight_layout()
fig2.show()

#%%
# ── 3. Sort by eROAS descending ────────────────────────────────────────────────

eroas_q95 = df["eROAS"].quantile(value_filtering_quantile)
quality_q95 = df["best_quality_score"].quantile(value_filtering_quantile)
conversion_q95 = df["best_conversion_prob"].quantile(value_filtering_quantile)

df_eroas_filtered = df[df["eROAS"] <= eroas_q95].reset_index(drop=True)
df_quality_score_filtered = df[df["best_quality_score"] <= quality_q95].reset_index(drop=True)
df_conversion_prob_filtered = df[df["best_conversion_prob"] <= conversion_q95].reset_index(drop=True)


print(f"eROAS 0.95 quantile threshold:         {eroas_q95:.4f}")
print(f"best_quality_score 0.95 quantile threshold: {quality_q95:.4f}")
print(f"best_conversion_prob 0.95 quantile threshold: {conversion_q95:.4f}")
print(f"df_eroas_filtered:          {len(df_eroas_filtered):,} rows (removed {len(df) - len(df_eroas_filtered):,})")
print(f"df_quality_score_filtered:  {len(df_quality_score_filtered):,} rows (removed {len(df) - len(df_quality_score_filtered):,})")
print(f"df_conversion_prob_filtered:  {len(df_conversion_prob_filtered):,} rows (removed {len(df) - len(df_conversion_prob_filtered):,})")



#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_eroas_filtered = df_eroas_filtered.sort_values("eROAS", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_eroas_filtered.iterrows():
    if pd.isna(row["eROAS"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append(row["occurred_at"])
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for ts in best_opportunities:
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(ts)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1                                     
bar_width = 50                                                                          
print("\nHourly distribution of best eROAS:")
for hour in range(24):
    count = len(hourly_buckets[hour])
    filled = round(count / max_count * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  {hour:02d}h  {count:4d}  {bar}")


hours = list(range(24))                                                                                                  
eroas_counts = [len(hourly_buckets[h]) for h in hours]

fig_e, axes_e = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig_e.suptitle("Hourly Distribution of Best eROAS Opportunities")

for ax, log in zip(axes_e, [False, True]):
    ax.bar(hours, eroas_counts, color="steelblue", edgecolor="white")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count (log scale)" if log else "Count")
    ax.set_title("Log Scale" if log else "Original Scale")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if log:
        ax.set_yscale("log")
    for i, v in enumerate(eroas_counts):
        if v > 0:
            ax.text(i, v * (1.05 if log else 1) + (0 if log else max(eroas_counts) * 0.01),
                    str(v), ha="center", va="bottom", fontsize=6)

fig_e.tight_layout()
fig_e.show()

#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_quality_score_filtered = df_quality_score_filtered.sort_values("best_quality_score", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_quality_score_filtered.iterrows():
    if pd.isna(row["best_quality_score"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append(row["occurred_at"])
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for ts in best_opportunities:
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(ts)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1                                     
bar_width = 50
print("\nHourly distribution of best Pclick:")
for hour in range(24):
    count = len(hourly_buckets[hour])
    filled = round(count / max_count * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  {hour:02d}h  {count:4d}  {bar}")

hours = list(range(24))
pclick_counts = [len(hourly_buckets[h]) for h in hours]

fig_p, axes_p = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig_p.suptitle("Hourly Distribution of Best Pclick Opportunities")

for ax, log in zip(axes_p, [False, True]):
    ax.bar(hours, pclick_counts, color="darkorange", edgecolor="white")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count (log scale)" if log else "Count")
    ax.set_title("Log Scale" if log else "Original Scale")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if log:
        ax.set_yscale("log")
    for i, v in enumerate(pclick_counts):
        if v > 0:
            ax.text(i, v * (1.05 if log else 1) + (0 if log else max(pclick_counts) * 0.01),
                    str(v), ha="center", va="bottom", fontsize=6)

fig_p.tight_layout()
fig_p.show()


#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_conversion_prob_filtered = df_conversion_prob_filtered.sort_values("best_conversion_prob", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_conversion_prob_filtered.iterrows():
    if pd.isna(row["best_conversion_prob"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append(row["occurred_at"])
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for ts in best_opportunities:
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(ts)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1                                     
bar_width = 50
print("\nHourly distribution of best Conversion:")
for hour in range(24):
    count = len(hourly_buckets[hour])
    filled = round(count / max_count * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  {hour:02d}h  {count:4d}  {bar}")

hours = list(range(24))
pclick_counts = [len(hourly_buckets[h]) for h in hours]

fig_p, axes_p = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig_p.suptitle("Hourly Distribution of Best Conversion Opportunities")

for ax, log in zip(axes_p, [False, True]):
    ax.bar(hours, pclick_counts, color="darkorange", edgecolor="white")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count (log scale)" if log else "Count")
    ax.set_title("Log Scale" if log else "Original Scale")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if log:
        ax.set_yscale("log")
    for i, v in enumerate(pclick_counts):
        if v > 0:
            ax.text(i, v * (1.05 if log else 1) + (0 if log else max(pclick_counts) * 0.01),
                    str(v), ha="center", va="bottom", fontsize=6)

fig_p.tight_layout()
fig_p.show()
