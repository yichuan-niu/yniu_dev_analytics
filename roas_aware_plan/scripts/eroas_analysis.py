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
daily_budget = 42500 # in cent
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
    - If target IS the winner → eROAS = winner_epv / winner_impression_cost
    - If target is NOT the winner → for each target candidate compute
      epv / impression_cost, then take the maximum.
    """
    # --- Target campaign entries ---
    target_entries = [c for c in candidates if c.get("campaignId") == target_campaign_id]
    if not target_entries:
        return {"eROAS": np.nan, "impression_cost": np.nan, "best_quality_score": np.nan}

    def epv(c):
        return (c.get("itemPrice") or 0.0) * (c.get("adQualityScore") or 0.0) * (c.get("predictedCtc") or 0.0)

    best_quality_score = max(c.get("adQualityScore") or 0.0 for c in target_entries)
    best_conversion_prob = max((c.get("adQualityScore") or 0.0) * (c.get("predictedCtc") or 0.0) for c in target_entries)

    # --- Winner entry ---
    winner_entries = [c for c in candidates if c.get("auctionRank") == 0]
    if not winner_entries:
        return {"eROAS": np.nan, "impression_cost": np.nan, "best_quality_score": best_quality_score}
    winner = winner_entries[0]

    # --- eROAS ---
    if winner.get("campaignId") == target_campaign_id:
        # Target won: use the winner entry's EPV and nextAdScore as impression cost
        pricing = winner.get("pricingMetadata") or {}
        impression_cost = pricing.get("nextAdScore", np.nan)
        eroas = epv(winner) / impression_cost if impression_cost and impression_cost > 0 else np.nan
    else:
        # Target didn't win: impression_cost is the winner's adScore (cost to beat)
        impression_cost = winner.get("adScore", np.nan)
        if impression_cost and impression_cost > 0:
            eroas = max(epv(c) / impression_cost for c in target_entries)
        else:
            eroas = np.nan

    return {
        "eROAS": eroas,
        "epv": eroas * impression_cost,
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

eroas_boundaries = eroas.quantile(q_levels).values
eroas_quantile_values = eroas.quantile(quantile_levels)
bqs_quantile_values = bqs.quantile(quantile_levels)

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
epv_q95 = df["epv"].quantile(value_filtering_quantile)

df_eroas_filtered = df[df["eROAS"] <= eroas_q95].reset_index(drop=True)
df_quality_score_filtered = df[df["best_quality_score"] <= quality_q95].reset_index(drop=True)
df_conversion_prob_filtered = df[df["best_conversion_prob"] <= conversion_q95].reset_index(drop=True)
df_epv_filtered = df[df["epv"] <= epv_q95].reset_index(drop=True)

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
    best_opportunities.append({
        "occurred_at": row["occurred_at"],
        "eROAS": row["eROAS"],
        "impression_cost": row["impression_cost"],
    })
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for opp in best_opportunities:
    ts = opp["occurred_at"]
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(opp)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1
bar_width = 50


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

# ── Figure: Hourly eROAS from best opportunities ───────────────────────────────

def hourly_eroas(buckets: dict) -> list:
    """For each hour, compute sum(epv) / sum(impression_cost) where epv = eROAS * impression_cost."""
    result = []
    for h in range(24):
        opps = buckets[h]
        total_cost = sum(o["impression_cost"] for o in opps)
        total_epv  = sum(o["eROAS"] * o["impression_cost"] for o in opps)
        result.append(total_epv / total_cost if total_cost > 0 else 0.0)
    return result

eroas_hourly = hourly_eroas(hourly_buckets)

fig_er, ax_er = plt.subplots(figsize=(14, 4))
fig_er.suptitle(f"Hourly eROAS by Distribution (Best eROAS Opportunities), Daily Budget = ${daily_budget / 100}")
ax_er.bar(hours, eroas_hourly, color="steelblue", edgecolor="white")
ax_er.set_xticks(hours)
ax_er.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax_er.set_xlabel("Hour")
ax_er.set_ylabel("eROAS")
ax_er.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(eroas_hourly):
    if v > 0:
        ax_er.text(i, v + max(eroas_hourly) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
fig_er.tight_layout()
fig_er.show()


#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_epv_filtered = df_epv_filtered.sort_values("epv", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_epv_filtered.iterrows():
    if pd.isna(row["eROAS"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append({
        "occurred_at": row["occurred_at"],
        "eROAS": row["eROAS"],
        "epv": row["epv"],
        "impression_cost": row["impression_cost"],
    })
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for opp in best_opportunities:
    ts = opp["occurred_at"]
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(opp)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1
bar_width = 50


hours = list(range(24))
eroas_counts = [len(hourly_buckets[h]) for h in hours]

fig_e, axes_e = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig_e.suptitle("Hourly Distribution of Best ePV Opportunities")

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

# ── Figure: Hourly eROAS from best opportunities ───────────────────────────────

def hourly_eroas(buckets: dict) -> list:
    """For each hour, compute sum(epv) / sum(impression_cost) where epv = eROAS * impression_cost."""
    result = []
    for h in range(24):
        opps = buckets[h]
        total_cost = sum(o["impression_cost"] for o in opps)
        total_epv  = sum(o["epv"] for o in opps)
        result.append(total_epv / total_cost if total_cost > 0 else 0.0)
    return result

eroas_hourly = hourly_eroas(hourly_buckets)

fig_er, ax_er = plt.subplots(figsize=(14, 4))
fig_er.suptitle(f"Hourly eROAS by Distribution (Best ePV Opportunities), Daily Budget = ${daily_budget / 100}")
ax_er.bar(hours, eroas_hourly, color="steelblue", edgecolor="white")
ax_er.set_xticks(hours)
ax_er.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax_er.set_xlabel("Hour")
ax_er.set_ylabel("eROAS")
ax_er.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(eroas_hourly):
    if v > 0:
        ax_er.text(i, v + max(eroas_hourly) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
fig_er.tight_layout()
fig_er.show()

#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_quality_score_filtered = df_quality_score_filtered.sort_values("best_quality_score", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_quality_score_filtered.iterrows():
    if pd.isna(row["best_quality_score"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append({
        "occurred_at": row["occurred_at"],
        "eROAS": row["eROAS"],
        "impression_cost": row["impression_cost"],
    })
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for opp in best_opportunities:
    ts = opp["occurred_at"]
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(opp)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1
bar_width = 50

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

# ── Figure: Hourly eROAS from best Pclick opportunities ───────────────────────

pclick_eroas_hourly = hourly_eroas(hourly_buckets)

fig_pr, ax_pr = plt.subplots(figsize=(14, 4))
fig_pr.suptitle(f"Hourly eROAS by Distribution (Best Pclick Opportunities), Daily Budget = ${daily_budget / 100}")
ax_pr.bar(hours, pclick_eroas_hourly, color="darkorange", edgecolor="white")
ax_pr.set_xticks(hours)
ax_pr.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax_pr.set_xlabel("Hour")
ax_pr.set_ylabel("eROAS")
ax_pr.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(pclick_eroas_hourly):
    if v > 0:
        ax_pr.text(i, v + max(pclick_eroas_hourly) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
fig_pr.tight_layout()
fig_pr.show()


#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_conversion_prob_filtered = df_conversion_prob_filtered.sort_values("best_conversion_prob", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_conversion_prob_filtered.iterrows():
    if pd.isna(row["best_conversion_prob"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append({
        "occurred_at": row["occurred_at"],
        "eROAS": row["eROAS"],
        "impression_cost": row["impression_cost"],
    })
    ad_spend += row["impression_cost"]
    if ad_spend > daily_budget:
        break

print(f"\nBest opportunities: {len(best_opportunities):,} auctions")
print(f"Cumulative ad_spend: {ad_spend:.4f}  (budget: {daily_budget})")


# ── 5. Distribute into 24 hourly buckets ──────────────────────────────────────

hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for opp in best_opportunities:
    ts = opp["occurred_at"]
    if pd.isna(ts):
        continue
    hourly_buckets[pd.Timestamp(ts).hour].append(opp)


max_count = max(len(v) for v in hourly_buckets.values()) if hourly_buckets else 1
bar_width = 50

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

# ── Figure: Hourly eROAS from best Conversion opportunities ───────────────────

conv_eroas_hourly = hourly_eroas(hourly_buckets)

fig_cr, ax_cr = plt.subplots(figsize=(14, 4))
fig_cr.suptitle(f"Hourly eROAS by Distribution (Best Conversion Opportunities), Daily Budget = ${daily_budget / 100}")
ax_cr.bar(hours, conv_eroas_hourly, color="darkorange", edgecolor="white")
ax_cr.set_xticks(hours)
ax_cr.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax_cr.set_xlabel("Hour")
ax_cr.set_ylabel("eROAS")
ax_cr.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(conv_eroas_hourly):
    if v > 0:
        ax_cr.text(i, v + max(conv_eroas_hourly) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
fig_cr.tight_layout()
fig_cr.show()


#%%
# ── Prod Hourly eROAS by Distribution ──

# ── Win count buckets: traverse all auctions in df ────────────────────────────
prod_win_buckets: dict[int, int] = {hour: 0 for hour in range(24)}

for _, row in df.iterrows():
    if pd.isna(row["occurred_at"]):
        continue
    candidates = row["candidates"]
    winner_entries = [c for c in candidates if c.get("auctionRank") == 0 and c.get("campaignId") == target_campaign_id]
    if not winner_entries:
        continue
    prod_win_buckets[pd.Timestamp(row["occurred_at"]).hour] += 1

prod_win_counts = [prod_win_buckets[h] for h in hours]

max_count = max(prod_win_counts) if max(prod_win_counts) > 0 else 1
bar_width = 50

fig_pw, axes_pw = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig_pw.suptitle("Hourly Distribution of Prod Auction Wins")

for ax, log in zip(axes_pw, [False, True]):
    ax.bar(hours, prod_win_counts, color="seagreen", edgecolor="white")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count (log scale)" if log else "Count")
    ax.set_title("Log Scale" if log else "Original Scale")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if log:
        ax.set_yscale("log")
    for i, v in enumerate(prod_win_counts):
        if v > 0:
            ax.text(i, v * (1.05 if log else 1) + (0 if log else max(prod_win_counts) * 0.01),
                    str(v), ha="center", va="bottom", fontsize=6)

fig_pw.tight_layout()
fig_pw.show()

# ── eROAS buckets: traverse df_eroas_filtered (outliers removed) ──────────────
prod_hourly_buckets: dict[int, list] = {hour: [] for hour in range(24)}

for _, row in df_eroas_filtered.iterrows():
    if pd.isna(row["eROAS"]) or pd.isna(row["impression_cost"]) or pd.isna(row["occurred_at"]):
        continue
    candidates = row["candidates"]
    winner_entries = [c for c in candidates if c.get("auctionRank") == 0 and c.get("campaignId") == target_campaign_id]
    if not winner_entries:
        continue
    hour = pd.Timestamp(row["occurred_at"]).hour
    prod_hourly_buckets[hour].append({
        "eROAS": row["eROAS"],
        "epv": row["epv"],
        "impression_cost": row["impression_cost"],
    })

prod_eroas_hourly = hourly_eroas(prod_hourly_buckets)

fig_prod, ax_prod = plt.subplots(figsize=(14, 4))
fig_prod.suptitle(f"Hourly eROAS by Distribution (Prod), Daily Budget = ${daily_budget / 100}")
ax_prod.bar(hours, prod_eroas_hourly, color="seagreen", edgecolor="white")
ax_prod.set_xticks(hours)
ax_prod.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax_prod.set_xlabel("Hour")
ax_prod.set_ylabel("eROAS")
ax_prod.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(prod_eroas_hourly):
    if v > 0:
        ax_prod.text(i, v + max(prod_eroas_hourly) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
fig_prod.tight_layout()
fig_prod.show()


#%%
# ── Traffic share by hour ───────────────────────────────────────────────────────

df_traffic = pd.read_csv(f"data/traffic_cmp_{target_campaign_id}_2026_03_21.csv")

fig_tr, ax_tr = plt.subplots(figsize=(14, 4))
fig_tr.suptitle("Hourly Traffic Share")
ax_tr.bar(df_traffic["utc_hour"], df_traffic["hourly_traffic_share"], color="steelblue", edgecolor="white")
ax_tr.set_xticks(df_traffic["utc_hour"])
ax_tr.set_xticklabels([f"{h:02d}h" for h in df_traffic["utc_hour"]], rotation=45, ha="right", fontsize=8)
ax_tr.set_xlabel("Hour of Day")
ax_tr.set_ylabel("Hourly Traffic Share")
ax_tr.grid(True, axis="y", linestyle="--", alpha=0.5)
for _, row in df_traffic.iterrows():
    if row["hourly_traffic_share"] > 0:
        ax_tr.text(row["utc_h2our"], row["hourly_traffic_share"] + df_traffic["hourly_traffic_share"].max() * 0.01,
                   f"{row['hourly_traffic_share']:.3f}", ha="center", va="bottom", fontsize=6)
fig_tr.tight_layout()
fig_tr.show()
