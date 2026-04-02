import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvr_map import dd_sic_organic_cvr
plt.close("all")

# ── Config ─────────────────────────────────────────────────────────────────────

target_campaign_id = "0b76b55d-a017-4f77-a9a7-38fc41c90d2d"
daily_budget = 42500 # in cent
revenue_guardrail = 0.03


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
        "iepv": ieroas * impression_cost,
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

eroas = df["ieROAS"].dropna()
bqs = df["best_quality_score"].dropna()

eroas_boundaries = eroas.quantile(q_levels).values
eroas_quantile_values = eroas.quantile(quantile_levels)
bqs_quantile_values = bqs.quantile(quantile_levels)

# ── Figure 1: quantile vs value for eROAS and best_quality_score ────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 5))

axes1[0].plot(quantile_levels, eroas_quantile_values, marker="o", markersize=4, color="steelblue", linewidth=2)
axes1[0].set_yscale("log")
axes1[0].set_xlabel("Quantile")
axes1[0].set_ylabel("ieROAS (log scale)")
axes1[0].set_title("ieROAS by Quantile (0.05 increments)")
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
df_valid = df[df["ieROAS"].notna() & df["best_quality_score"].notna()].copy()
df_valid["ieroas_bucket"] = pd.cut(
    df_valid["ieROAS"],
    bins=eroas_boundaries,
    labels=[f"{v:.2f}" for v in eroas_quantile_values],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("ieroas_bucket", observed=True)["best_quality_score"].mean()

fig2, ax2 = plt.subplots(figsize=(14, 5))
bars = ax2.bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2.set_xticks(range(len(avg_bqs)))
ax2.set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2.set_xlabel("ieROAS (quantile bucket upper bound)")
ax2.set_ylabel("Avg best_quality_score")
ax2.set_title("Avg best_quality_score per ieROAS Quantile Bucket (0.05 increments)")
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)

fig2.tight_layout()
fig2.show()


# ── Figure 3: avg best_quality_score per eROAS quantile bucket ──────────────
df_valid = df[df["ieROAS"].notna() & df["best_conversion_prob"].notna()].copy()
df_valid["ieroas_bucket"] = pd.cut(
    df_valid["ieROAS"],
    bins=eroas_boundaries,
    labels=[f"{v:.2f}" for v in eroas_quantile_values],
    include_lowest=True,
)
avg_bqs = df_valid.groupby("ieroas_bucket", observed=True)["best_conversion_prob"].mean()

fig2, ax2 = plt.subplots(figsize=(14, 5))
bars = ax2.bar(range(len(avg_bqs)), avg_bqs.values, color="steelblue", edgecolor="white")
ax2.set_xticks(range(len(avg_bqs)))
ax2.set_xticklabels(avg_bqs.index, rotation=45, ha="right", fontsize=7)
ax2.set_xlabel("ieROAS (quantile bucket upper bound)")
ax2.set_ylabel("Avg best_conversion_prob")
ax2.set_title("Avg best_conversion_prob per ieROAS Quantile Bucket (0.05 increments)")
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, avg_bqs.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_bqs.values) * 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=6)

fig2.tight_layout()
fig2.show()

