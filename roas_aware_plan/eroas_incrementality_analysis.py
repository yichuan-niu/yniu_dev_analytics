import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvr_map import dd_sic_organic_cvr
plt.close("all")

# ── Config ─────────────────────────────────────────────────────────────────────

target_campaign_id = "0b76b55d-a017-4f77-a9a7-38fc41c90d2d"
daily_budget = 42500 # in cent
revenue_guardrail = 0.00
value_filtering_quantile = 0.95


#%%
# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_pickle(f"data/auction_history_cmp_{target_campaign_id}.pkl")
print(f"Loaded {len(df):,} auctions")

#%%
# ── 2. Compute per-auction metrics ─────────────────────────────────────────────

def compute_metrics(candidates: list, target_campaign_id: str) -> dict:
    # --- Target campaign entries ---
    target_entries = [c for c in candidates if c.get("campaignId") == target_campaign_id]
    if not target_entries:
        return {"eROAS": np.nan, "impression_cost": np.nan, "best_quality_score": np.nan}

    def iepv(c):
        return ((
            c.get("itemPrice") or 0.0) * 
            max(
                ((c.get("adQualityScore") or 0.0) * (c.get("predictedCtc") or 0.0) 
                 - (dd_sic_organic_cvr.get(c.get("ddSic") or "") or 0.0)),
                revenue_guardrail
            )
        )

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
        ieroas = iepv(winner) / impression_cost if impression_cost and impression_cost > 0 else np.nan
    else:
        # Target didn't win: impression_cost is the winner's adScore (cost to beat)
        impression_cost = winner.get("adScore", np.nan)
        if impression_cost and impression_cost > 0:
            ieroas = max(iepv(c) / impression_cost for c in target_entries)
        else:
            ieroas = np.nan

    return {
        "ieROAS": ieroas,
        "iePV": ieroas * impression_cost,
        "impression_cost": impression_cost,
        "best_quality_score": best_quality_score,
        "best_conversion_prob": best_conversion_prob
    }


metrics = df["candidates"].apply(
    lambda cands: pd.Series(compute_metrics(cands, target_campaign_id))
)
df = pd.concat([df, metrics], axis=1)

print(f"ieROAS computed: {df['ieROAS'].notna().sum():,} valid / {len(df):,} total auctions")


#%%

q_levels = np.arange(0.0, 1.01, 0.05)
quantile_levels = np.arange(0.05, 1.01, 0.05)

ieroas = df["ieROAS"].dropna()
iepv = df["iePV"].dropna()

ieroas_boundaries = np.unique(ieroas.quantile(q_levels).values)
ieroas_quantile_values = ieroas.quantile(quantile_levels)

iepv_boundaries = np.unique(iepv.quantile(q_levels).values)
iepv_quantile_values = iepv.quantile(quantile_levels)

# ── Figure 1: quantile vs value for eROAS and best_quality_score ────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 5))

axes1[0].plot(quantile_levels, ieroas_quantile_values, marker="o", markersize=4, color="steelblue", linewidth=2)
axes1[0].set_yscale("log")
axes1[0].set_xlabel("Quantile")
axes1[0].set_ylabel("ieROAS (log scale)")
axes1[0].set_title("ieROAS by Quantile (0.05 increments)")
axes1[0].set_xticks(quantile_levels)
axes1[0].set_xticklabels([f"{q:.2f}" for q in quantile_levels], rotation=45, ha="right", fontsize=7)
axes1[0].grid(True, linestyle="--", alpha=0.5)
for q, v in zip(quantile_levels, ieroas_quantile_values):
    axes1[0].annotate(f"{v:.2f}", xy=(q, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)

axes1[1].plot(quantile_levels, iepv_quantile_values, marker="o", markersize=4, color="steelblue", linewidth=2)
axes1[1].set_yscale("log")
axes1[1].set_xlabel("Quantile")
axes1[1].set_ylabel("iePV $ (log scale)")
axes1[1].set_title("iePV by Quantile (0.05 increments)")
axes1[1].set_xticks(quantile_levels)
axes1[1].set_xticklabels([f"{q:.2f}" for q in quantile_levels], rotation=45, ha="right", fontsize=7)
axes1[1].grid(True, linestyle="--", alpha=0.5)
for q, v in zip(quantile_levels, iepv_quantile_values):
    axes1[1].annotate(f"{v:.2f}", xy=(q, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)

fig1.tight_layout()
fig1.show()

# ── Figure 2: avg best_quality_score per eROAS quantile bucket ──────────────
df_valid = df[df["ieROAS"].notna() & df["best_quality_score"].notna()].copy()
df_valid["ieROAS_bucket"] = pd.cut(
    df_valid["ieROAS"],
    bins=np.unique(ieroas_boundaries),
    labels=[f"{v:.2f}" for v in ieroas_boundaries[ieroas_boundaries > 0]],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("ieROAS_bucket", observed=True)["best_quality_score"].mean()



fig2, ax2 = plt.subplots(1, 2, figsize=(18, 5))
bars = ax2[0].bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2[0].set_xticks(range(len(avg_bqs)))
ax2[0].set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2[0].set_xlabel("ieROAS (quantile bucket upper bound)")
ax2[0].set_ylabel("Avg best_quality_score")
ax2[0].set_title("Avg best_quality_score per ieROAS Quantile Bucket (0.05 increments)")
ax2[0].grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)


df_valid["iePV_bucket"] = pd.cut(
    df_valid["iePV"],
    bins=np.unique(iepv_boundaries),
    labels=[f"{v:.2f}" for v in iepv_boundaries[iepv_boundaries > 0]],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("iePV_bucket", observed=True)["best_quality_score"].mean()
bars = ax2[1].bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2[1].set_xticks(range(len(avg_bqs)))
ax2[1].set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2[1].set_xlabel("iePV (quantile bucket upper bound)")
ax2[1].set_ylabel("Avg best_quality_score")
ax2[1].set_title("Avg best_quality_score per iePV Quantile Bucket (0.05 increments)")
ax2[1].grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)


fig2.tight_layout()
fig2.show()


# ── Figure 3: avg best_quality_score per eROAS quantile bucket ──────────────
df_valid = df[df["ieROAS"].notna() & df["best_conversion_prob"].notna()].copy()
df_valid["ieROAS_bucket"] = pd.cut(
    df_valid["ieROAS"],
    bins=np.unique(ieroas_boundaries),
    labels=[f"{v:.2f}" for v in ieroas_boundaries[ieroas_boundaries > 0]],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("ieROAS_bucket", observed=True)["best_conversion_prob"].mean()

fig2, ax2 = plt.subplots(1, 2, figsize=(18, 5))
bars = ax2[0].bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2[0].set_xticks(range(len(avg_bqs)))
ax2[0].set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2[0].set_xlabel("ieROAS (quantile bucket upper bound)")
ax2[0].set_ylabel("Avg best_conversion_prob")
ax2[0].set_title("Avg best_conversion_prob per ieROAS Quantile Bucket (0.05 increments)")
ax2[0].grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)

df_valid["iePV_bucket"] = pd.cut(
    df_valid["iePV"],
    bins=np.unique(iepv_boundaries),
    labels=[f"{v:.2f}" for v in iepv_boundaries[iepv_boundaries > 0]],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("iePV_bucket", observed=True)["best_conversion_prob"].mean()
bars = ax2[1].bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2[1].set_xticks(range(len(avg_bqs)))
ax2[1].set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2[1].set_xlabel("iePV (quantile bucket upper bound)")
ax2[1].set_ylabel("Avg best_conversion_prob")
ax2[1].set_title("Avg best_conversion_prob per iePV Quantile Bucket (0.05 increments)")
ax2[1].grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)

fig2.tight_layout()
fig2.show()

#%%


# ── 3. Sort by eROAS descending ────────────────────────────────────────────────

eroas_q95 = df["ieROAS"].quantile(value_filtering_quantile)
epv_q95 = df["iePV"].quantile(value_filtering_quantile)

df_ieroas_filtered = df[df["ieROAS"] <= eroas_q95].reset_index(drop=True)
df_epv_filtered = df[df["iePV"] <= epv_q95].reset_index(drop=True)

print(f"df_eroas_filtered: {len(df_ieroas_filtered):,} rows (removed {len(df) - len(df_ieroas_filtered):,})")


#%%
# ── 4. Collect best opportunities within daily budget ──────────────────────────
df_ieroas_filtered = df_ieroas_filtered.sort_values("ieROAS", ascending=False, na_position="last").reset_index(drop=True)


best_opportunities = []
ad_spend = 0.0

for _, row in df_ieroas_filtered.iterrows():
    if pd.isna(row["ieROAS"]) or pd.isna(row["impression_cost"]):
        continue
    best_opportunities.append({
        "occurred_at": row["occurred_at"],
        "ieROAS": row["ieROAS"],
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

def hourly_ieroas(buckets: dict) -> list:
    result = []
    for h in range(24):
        opps = buckets[h]
        total_cost = sum(o["impression_cost"] for o in opps)
        total_epv  = sum(o["ieROAS"] * o["impression_cost"] for o in opps)
        result.append(total_epv / total_cost if total_cost > 0 else 0.0)
    return result

eroas_hourly = hourly_ieroas(hourly_buckets)

fig_e, axes_e = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig_e.suptitle("Hourly Distribution of Best ieROAS Opportunities")

ax = axes_e[0]
ax.bar(hours, eroas_counts, color="steelblue", edgecolor="white")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Hour")
ax.set_ylabel("Count")
ax.set_title("Original Scale")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(eroas_counts):
    if v > 0:
        ax.text(i, v + max(eroas_counts) * 0.01, str(v), ha="center", va="bottom", fontsize=6)

ax = axes_e[1]
ax.bar(hours, eroas_hourly, color="steelblue", edgecolor="white")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Hour")
ax.set_ylabel("ieROAS")
ax.set_title(f"Hourly ieROAS by Distribution (Best ieROAS Opportunities), Daily Budget = ${daily_budget / 100}")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(eroas_hourly):
    if v > 0:
        ax.text(i, v + max(eroas_hourly) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)

fig_e.tight_layout()
fig_e.show()
