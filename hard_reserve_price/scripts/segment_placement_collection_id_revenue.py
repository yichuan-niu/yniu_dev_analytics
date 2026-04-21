import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector

plt.close("all")

# ── Constants ──────────────────────────────────────────────────────────────────
EVENT_DATE             = "2026-03-25"
SAMPLE_PCT             = 100   # campaign-level sampling
MAX_RESERVE_INCREMENT  = 5.0
MIN_COLLECTION_CLICKS  = 50   # skip collection-placement cohorts with fewer clicks than this threshold
ROAS_SNAPSHOT_START    = "2026-03-19"
ROAS_SNAPSHOT_END      = "2026-03-25"

PLACEMENT_GROUPS = {
    "Search": [
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_SEARCH",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_GLOBAL_SEARCH",
    ],
    "Category": [
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_CATEGORY_L1",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_CATEGORY_L2",
    ],
    "Collection": [
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_COLLECTION",
    ],
    "DoubleDash": [
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_COLLECTION",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_CATEGORY_L1",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_CATEGORY_L2",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_SEARCH",
    ],
}
PLACEMENT_TO_GROUP = {p: g for g, ps in PLACEMENT_GROUPS.items() for p in ps}

# Each placement group uses a different secondary cohort dimension.
# None means the placement group itself is the only dimension (no sub-cohort).
COHORT_DIM = {
    "Search":     "normalized_query",
    "Category":   "l1_category_id",
    "Collection": "collection_id",
    "DoubleDash": None,
}
PLACEMENT_GROUP_ORDER = ["Search", "Category", "Collection", "DoubleDash"]


# ── Snowflake connection ───────────────────────────────────────────────────────
def get_connection() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse="TEAM_ADS_DEMAND_REPORTING_2XL",
        role=os.environ["SNOWFLAKE_ROLE"],
        database="edw",
        schema="ads",
    )


# ── Query ──────────────────────────────────────────────────────────────────────
# Uses collection_id directly from the auction table.
# NULL collection_id rows are labeled 'Unknown'.
QUERY = """
WITH winners AS (
    SELECT
        acd.auction_id,
        acd.campaign_id,
        acd.placement,
        acd.auction_bid / 100.0                                                     AS auction_bid_dollars,
        acd.bid_price_unit_amount / 100.0                                           AS cpc_dollars,
        GET(PARSE_JSON(acd.pricing_metadata), 'cpcGsp')::INT / 100.0               AS raw_gsp_dollars,
        GET(PARSE_JSON(acd.pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
        GET(PARSE_JSON(acd.pricing_metadata), 'softReserveBeta')::FLOAT
            * GET(PARSE_JSON(acd.pricing_metadata), 'nextBid')::INT / 100.0        AS soft_reserve_dollars,
        COALESCE(acd.normalized_query, 'Unknown')                                  AS normalized_query,
        COALESCE(acd.l1_category_id, 'Unknown')                                    AS l1_category_id,
        COALESCE(acd.collection_id, 'Unknown')                                     AS collection_id
    FROM edw.ads.ads_auction_candidates_event_delta acd
    WHERE acd.event_date = '{event_date}'
      AND acd.CURRENCY_ISO_TYPE IN ('USD')
      AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
      AND acd.auction_rank = 0
      AND acd.pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(acd.campaign_id)), 100) < {sample_pct}
),
clicked AS (
    SELECT
        ad_auction_id,
        MIN(event_timestamp) AS event_timestamp
    FROM proddb.public.fact_item_card_click_dedup
    WHERE event_date = '{event_date}'
      AND is_sponsored = 1
      AND is_cpc = 1
      AND ad_auction_id IS NOT NULL
      AND campaign_id IS NOT NULL
    GROUP BY ad_auction_id
)
SELECT
    winners.campaign_id,
    winners.placement,
    winners.normalized_query,
    winners.l1_category_id,
    winners.collection_id,
    auction_bid_dollars,
    cpc_dollars,
    raw_gsp_dollars,
    soft_reserve_dollars,
    hard_reserve_dollars,
    clicked.event_timestamp
FROM winners
INNER JOIN clicked ON winners.auction_id = clicked.ad_auction_id
"""

BUDGET_QUERY = """
SELECT
    campaign_id,
    COALESCE(MAX(campaign_budget), SUM(daily_budget)) / 100 AS campaign_daily_budget_dollars
FROM PRODDB.PUBLIC.FACT_ADS_DAILY_BUDGET
WHERE date_est = '{budget_date}'
GROUP BY campaign_id
"""

# Returns total attributed sales and total ad fee per campaign over a snapshot window.
# Used to compute before/after ROAS per (placement_group, cohort_key) cohort.
ROAS_QUERY = """
SELECT
    CAMPAIGN_ID,
    SUM(TOTAL_CX_SALES_AMOUNT_LOCAL) / 100.0  AS total_attributed_sales_usd,
    SUM(TOTAL_CX_AD_FEE_LOCAL) / 100.0        AS total_ad_fee_usd
FROM EDW.ADS.FACT_CPG_CPC_CAMPAIGN_PERFORMANCE
WHERE SNAPSHOT_DATE BETWEEN '{start_date}' AND '{end_date}'
  AND TIMEZONE_TYPE = 'utc'
  AND DAYPART_NAME = 'day'
  AND REPORT_TYPE   = '[brand_cohorts] campaign'
GROUP BY CAMPAIGN_ID
HAVING SUM(TOTAL_CX_AD_FEE_LOCAL) > 0
"""


def fetch_data(event_date: str = EVENT_DATE) -> pd.DataFrame:
    query = QUERY.format(sample_pct=SAMPLE_PCT, event_date=event_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    for col in ["auction_bid_dollars", "cpc_dollars", "raw_gsp_dollars",
                "soft_reserve_dollars", "hard_reserve_dollars"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df


def fetch_roas(
    start_date: str = ROAS_SNAPSHOT_START,
    end_date: str = ROAS_SNAPSHOT_END,
) -> pd.DataFrame:
    """Return per-campaign [campaign_id, total_attributed_sales_usd, total_ad_fee_usd]."""
    query = ROAS_QUERY.format(start_date=start_date, end_date=end_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["campaign_id"] = df["campaign_id"].astype(str)
    for col in ["total_attributed_sales_usd", "total_ad_fee_usd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["total_ad_fee_usd"])


def fetch_budget(budget_date: str = EVENT_DATE) -> pd.DataFrame:
    query = BUDGET_QUERY.format(budget_date=budget_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["campaign_daily_budget_dollars"] = pd.to_numeric(
        df["campaign_daily_budget_dollars"], errors="coerce"
    )
    return df.dropna(subset=["campaign_daily_budget_dollars"])


# ── Revenue lift computation (per segment) ─────────────────────────────────────
def compute_revenue_lift_segment(
    df_seg: pd.DataFrame,
    budget_map: dict,
    max_delta: float = 1.0,
) -> tuple:
    """
    Budget-aware total revenue lift for a pre-filtered segment DataFrame.

    For each hard reserve increment delta:
      new_hr = hard_reserve + delta
      - If auction_bid < new_hr: auction is not winnable → new_cpc = 0
      - Otherwise: new_cpc = min(auction_bid, max(raw_gsp, soft_reserve, new_hr))
      change = new_cpc - cpc

    Denominator: budget-constrained baseline CPC of this segment.
    Budget is applied per campaign in chronological click order.

    Returns (DataFrame with columns [delta, total_lift_pct], segment_total_cpc),
    or (None, 0) if segment is empty.
    """
    work = (
        df_seg
        .sort_values(["campaign_id", "event_timestamp"], na_position="last")
        .reset_index(drop=True)
    )

    cpc = work["cpc_dollars"].to_numpy(dtype=float)
    bid = work["auction_bid_dollars"].to_numpy(dtype=float)
    gsp = work["raw_gsp_dollars"].to_numpy(dtype=float)
    sr  = work["soft_reserve_dollars"].to_numpy(dtype=float)
    hr  = work["hard_reserve_dollars"].to_numpy(dtype=float)

    cmp_ids = work["campaign_id"].to_numpy()
    _, first_idx = np.unique(cmp_ids, return_index=True)
    last_idx = np.append(first_idx[1:], len(cmp_ids))
    campaign_budgets = np.array(
        [budget_map.get(c, float("inf")) for c in cmp_ids[first_idx]], dtype=float
    )

    # Segment-level baseline CPC (budget-constrained)
    total_cpc = 0.0
    for i, (start, end) in enumerate(zip(first_idx, last_idx)):
        total_cpc += min(float(cpc[start:end].sum()), campaign_budgets[i])

    if total_cpc == 0:
        return None, 0.0

    competitive_floor = np.maximum(gsp, sr)  # max(raw_gsp, soft_reserve), precomputed

    # Formula-predicted CPC at delta=0 (used as anchor so change=0 at delta=0).
    # Raw cpc from data may differ from this formula due to data edge cases; anchoring
    # to the formula baseline ensures the lift curve is correctly zero-referenced.
    cpc_baseline = np.where(
        bid < hr,
        0.0,
        np.minimum(bid, np.maximum(competitive_floor, hr)),
    )

    deltas = np.arange(0.0, max_delta + 0.01, 0.01)
    records = []
    for delta in deltas:
        new_hr = hr + delta
        new_cpc = np.where(
            bid < new_hr,
            0.0,                                              # auction lost
            np.minimum(bid, np.maximum(competitive_floor, new_hr)),  # new charge
        )
        change = new_cpc - cpc_baseline

        total_lift = 0.0
        for i, (start, end) in enumerate(zip(first_idx, last_idx)):
            funded = np.cumsum(new_cpc[start:end]) <= campaign_budgets[i]
            total_lift += change[start:end][funded].sum()

        records.append({
            "delta":          round(float(delta), 2),
            "total_lift_pct": total_lift / total_cpc * 100,
        })

    return pd.DataFrame(records), total_cpc


# ── Plot helpers ──────────────────────────────────────────────────────────────
TOP_N = 30  # top N cohorts per placement group for plotting


def _top_cohorts(summary: pd.DataFrame, pg: str, n: int = TOP_N) -> pd.DataFrame:
    """Return top-n cohorts for a placement group sorted by total_lift_pct descending."""
    return (
        summary[summary["placement_group"] == pg]
        .nlargest(n, "total_lift_pct")
        .reset_index(drop=True)
    )


def plot_revenue_lift(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    """4 subplots (one per placement group): horizontal bar chart of revenue lift + best delta."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    fig.suptitle(
        f"Revenue Lift by Cohort (top {TOP_N} per placement group)\n"
        f"(SP clicked winners, budget-aware, {event_date})",
        fontsize=15,
    )
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        top = _top_cohorts(summary, pg)
        if top.empty:
            ax.set_title(f"{pg}\n(no data)", fontsize=12)
            continue
        labels = top["cohort_key"].astype(str).tolist()[::-1]
        lifts  = top["total_lift_pct"].tolist()[::-1]
        deltas = top["best_delta"].tolist()[::-1]
        y_pos  = np.arange(len(labels))
        bars = ax.barh(y_pos, lifts, color="steelblue", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Revenue Lift (%)")
        dim_label = COHORT_DIM.get(pg, pg) or pg
        ax.set_title(f"{pg}\n(dim: {dim_label})", fontsize=12)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"Δ${d:.2f}", va="center", fontsize=7, color="darkred")
    plt.tight_layout()
    plt.show()


def plot_roas(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    """4 subplots: ROAS before vs after for top cohorts per placement group."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    fig.suptitle(
        f"ROAS Before vs After (top {TOP_N} per placement group)\n"
        f"(SP clicked winners, budget-aware, {event_date})",
        fontsize=15,
    )
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        top = _top_cohorts(summary, pg)
        top = top.dropna(subset=["roas_before"])
        if top.empty:
            ax.set_title(f"{pg}\n(no data)", fontsize=12)
            continue
        labels = top["cohort_key"].astype(str).tolist()[::-1]
        before = top["roas_before"].tolist()[::-1]
        after  = top["roas_after"].tolist()[::-1]
        y_pos  = np.arange(len(labels))
        h = 0.35
        ax.barh(y_pos - h / 2, before, h, label="Before", color="steelblue", edgecolor="white")
        ax.barh(y_pos + h / 2, after,  h, label="After",  color="coral",     edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("ROAS")
        ax.set_title(f"{pg}", fontsize=12)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cpc(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    """4 subplots: avg CPC before vs after for top cohorts per placement group."""
    s = summary.copy()
    s["avg_cpc_before"] = s["segment_cpc"] / s["n_rows"]
    s["avg_cpc_after"]  = s["segment_cpc"] * (1 + s["total_lift_pct"] / 100) / s["n_rows"]

    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    fig.suptitle(
        f"Avg CPC Before vs After (top {TOP_N} per placement group)\n"
        f"(SP clicked winners, budget-aware, {event_date})",
        fontsize=15,
    )
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        top = _top_cohorts(s, pg)
        if top.empty:
            ax.set_title(f"{pg}\n(no data)", fontsize=12)
            continue
        labels = top["cohort_key"].astype(str).tolist()[::-1]
        before = top["avg_cpc_before"].tolist()[::-1]
        after  = top["avg_cpc_after"].tolist()[::-1]
        y_pos  = np.arange(len(labels))
        h = 0.35
        ax.barh(y_pos - h / 2, before, h, label="Before", color="steelblue", edgecolor="white")
        ax.barh(y_pos + h / 2, after,  h, label="After",  color="coral",     edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Avg CPC ($)")
        ax.set_title(f"{pg}", fontsize=12)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
#%%
print("Fetching auction data from Snowflake (clicked winners only)...")
# df = fetch_data()
# df.to_pickle("data/segment_placement_collection_id_revenue_df.pkl")

df = pd.read_pickle("data/segment_placement_collection_id_revenue_df.pkl")
print(f"  Total clicked winners: {len(df):,}")
print(f"  Total CPC ($):         {df['cpc_dollars'].sum():,.2f}")

#%%
print(f"\nFetching ROAS data ({ROAS_SNAPSHOT_START} – {ROAS_SNAPSHOT_END})...")
# roas_df = fetch_roas()
# roas_df.to_pickle("data/segment_placement_collection_id_roas_df.pkl")

roas_df = pd.read_pickle("data/segment_placement_collection_id_roas_df.pkl")
print(f"  Campaigns with ROAS data: {len(roas_df):,}")

#%%
print(f"\nFetching campaign daily budgets for {EVENT_DATE}...")
# budget_df = fetch_budget()
# budget_df.to_pickle("data/segment_placement_collection_id_budget_df.pkl")

budget_df = pd.read_pickle("data/segment_placement_collection_id_budget_df.pkl")
budget_map = budget_df.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
print(f"  Campaigns with known budget: {len(budget_map):,}")

#%%
# Map placement strings to group names
df["placement_group"] = df["placement"].map(PLACEMENT_TO_GROUP).fillna("Other")

# Assign cohort_key per placement group's secondary dimension
conditions = [df["placement_group"] == pg for pg in PLACEMENT_GROUP_ORDER]
choices = [
    df[COHORT_DIM[pg]] if COHORT_DIM[pg] else pg
    for pg in PLACEMENT_GROUP_ORDER
]
df["cohort_key"] = np.select(conditions, choices, default="Other")

# Enumerate cohorts with enough clicks
cohorts = (
    df.groupby(["placement_group", "cohort_key"])
    .size()
    .reset_index(name="n_rows")
    .query("n_rows >= @MIN_COLLECTION_CLICKS")
    .sort_values(["placement_group", "cohort_key"])
    .reset_index(drop=True)
)
print(f"\n  Cohorts with >= {MIN_COLLECTION_CLICKS} clicks: {len(cohorts)}")

#%%
summary_rows = []
for _, cohort in cohorts.iterrows():
    pg = cohort["placement_group"]
    ck = cohort["cohort_key"]
    df_seg = df[(df["placement_group"] == pg) & (df["cohort_key"] == ck)]

    results, seg_cpc = compute_revenue_lift_segment(df_seg, budget_map, max_delta=MAX_RESERVE_INCREMENT)
    if results is None:
        continue

    best = results.loc[results["total_lift_pct"].idxmax()]
    summary_rows.append({
        "placement_group": pg,
        "cohort_key":      ck,
        "n_rows":          int(cohort["n_rows"]),
        "segment_cpc":     seg_cpc,
        "best_delta":      float(best["delta"]),
        "total_lift_pct":  float(best["total_lift_pct"]),
    })
    print(f"  [{pg} / {ck}]  best Δ=${best['delta']:.2f}  lift={best['total_lift_pct']:.4f}%")

#%%
summary = (
    pd.DataFrame(summary_rows)
    .sort_values("total_lift_pct", ascending=False)
    .reset_index(drop=True)
)
# Only retain cohorts with positive revenue lift for all downstream analysis and plots
# summary = summary[summary["total_lift_pct"] > 0].reset_index(drop=True)
print(f"  Cohorts with positive revenue lift: {len(summary)}")

print("\nRevenue Lift by (Placement Group, Cohort Key) — sorted by total lift:")
print(f"{'Placement Group':<15} {'Cohort Key':<40} {'Rows':>8} {'Best Δ ($)':>12} {'Total Lift (%)':>16}")
print("-" * 97)
for _, row in summary.iterrows():
    print(
        f"{row['placement_group']:<15} {str(row['cohort_key']):<40}"
        f" {row['n_rows']:>8,} {row['best_delta']:>12.2f} {row['total_lift_pct']:>16.4f}"
    )

# Aggregate: weighted total lift across all cohorts using each cohort's best delta.
# lift_dollars = total_lift_pct/100 * segment_cpc  →  sum and re-express as % of total CPC.
total_lift_dollars = (summary["total_lift_pct"] / 100 * summary["segment_cpc"]).sum()
total_cpc_all = summary["segment_cpc"].sum()
overall_lift_pct = total_lift_dollars / total_cpc_all * 100 if total_cpc_all > 0 else 0.0
print(f"\n{'─' * 97}")
print(f"Overall total revenue lift (best Δ per cohort): {overall_lift_pct:.4f}%"
      f"  (${total_lift_dollars:,.2f} lift on ${total_cpc_all:,.2f} total CPC)")

#%%
plot_revenue_lift(summary)

#%%
# ── ROAS before / after hard reserve increment ─────────────────────────────────
# For each cohort, attribute each campaign's sales/ad_fee proportionally by CPC
# fraction to avoid double-counting campaigns that appear in multiple cohorts.
# Both sales and ad_fee are divided by _n_roas_days to normalise to 1-day basis,
# aligning with lift_dollars which is already a single-day figure.

from datetime import date as _date
_n_roas_days = (_date.fromisoformat(ROAS_SNAPSHOT_END) - _date.fromisoformat(ROAS_SNAPSHOT_START)).days + 1
print(f"ROAS window: {ROAS_SNAPSHOT_START} – {ROAS_SNAPSHOT_END} ({_n_roas_days} days)")

roas_lookup = roas_df.set_index("campaign_id")

# Total CPC per campaign across ALL auction data (used as attribution denominator)
_campaign_total_cpc = df.groupby("campaign_id")["cpc_dollars"].sum()

roas_rows = []
for _, row in summary.iterrows():
    pg = row["placement_group"]
    ck = row["cohort_key"]

    cohort_df = df[(df["placement_group"] == pg) & (df["cohort_key"] == ck)]
    cohort_cpc_by_campaign = cohort_df.groupby("campaign_id")["cpc_dollars"].sum()
    fractions = (cohort_cpc_by_campaign / _campaign_total_cpc.reindex(cohort_cpc_by_campaign.index)).fillna(1.0)

    cohort_roas = roas_lookup.reindex(fractions.index).dropna(subset=["total_ad_fee_usd"])
    cohort_roas = cohort_roas.join(fractions.rename("fraction"), how="left").fillna({"fraction": 1.0})

    if cohort_roas.empty or (cohort_roas["total_ad_fee_usd"] * cohort_roas["fraction"]).sum() == 0:
        continue

    cohort_sales  = (cohort_roas["total_attributed_sales_usd"] * cohort_roas["fraction"]).sum() / _n_roas_days
    cohort_ad_fee = (cohort_roas["total_ad_fee_usd"]           * cohort_roas["fraction"]).sum() / _n_roas_days
    lift_dollars  = row["total_lift_pct"] / 100 * row["segment_cpc"]

    roas_before = cohort_sales / cohort_ad_fee
    roas_after  = cohort_sales / (cohort_ad_fee + lift_dollars) if (cohort_ad_fee + lift_dollars) > 0 else 0.0

    roas_rows.append({
        "placement_group": pg,
        "cohort_key":      ck,
        "roas_before":     round(roas_before, 4),
        "roas_after":      round(roas_after,  4),
        "roas_change":     round(roas_after - roas_before, 4),
    })

roas_summary = pd.DataFrame(roas_rows)
summary = summary.merge(roas_summary, on=["placement_group", "cohort_key"], how="left")

print("\nROAS Before vs After (best Δ per cohort):")
print(f"{'Placement Group':<15} {'Cohort Key':<40} {'Best Δ ($)':>12} {'ROAS Before':>13} {'ROAS After':>12} {'ROAS Δ':>10}")
print("-" * 107)
for _, row in summary.sort_values("roas_before", ascending=False).iterrows():
    if pd.isna(row.get("roas_before")):
        continue
    print(
        f"{row['placement_group']:<15} {str(row['cohort_key']):<40}"
        f" {row['best_delta']:>12.2f} {row['roas_before']:>13.4f}"
        f" {row['roas_after']:>12.4f} {row['roas_change']:>10.4f}"
    )

#%%
plot_roas(summary)

#%%
plot_cpc(summary)

#%%
# ── Top 2 cohorts per placement group: lift curves ────────────────────────────
SEGMENT_MAX_DELTA = 2.0

fig, axes = plt.subplots(1, 4, figsize=(36, 6), sharey=False)
fig.suptitle(f"Revenue Lift vs Hard Reserve Increment (top 2 cohorts per group)\n"
             f"(SP clicked winners, budget-aware, {EVENT_DATE})", fontsize=14)

for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
    top2 = _top_cohorts(summary, pg, n=2)
    for _, row in top2.iterrows():
        ck = row["cohort_key"]
        df_seg = df[(df["placement_group"] == pg) & (df["cohort_key"] == ck)]
        results, _ = compute_revenue_lift_segment(df_seg, budget_map, max_delta=SEGMENT_MAX_DELTA)
        if results is None:
            continue
        ax.plot(results["delta"], results["total_lift_pct"], linewidth=2, label=str(ck))
    ax.set_xlabel("Hard Reserve Increment (Δ, $)", fontsize=12)
    ax.set_ylabel("Revenue Lift (%)", fontsize=12)
    ax.set_title(pg, fontsize=13)
    ax.set_xticks(np.arange(0, SEGMENT_MAX_DELTA + 0.1, 0.2))
    ax.set_xlim(0, SEGMENT_MAX_DELTA)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Cohort Key", fontsize=7)

plt.tight_layout()
plt.show()


#%%
# ── Debug: detailed breakdown for a cohort ────────────────────────────────────
def debug_cohort(pg: str, cohort_keys: list) -> None:
    """Print detailed metrics for specific cohorts within a placement group."""
    _roas_lookup = roas_df.set_index("campaign_id")
    cols = (
        f"{'Cohort Key':<40} {'Sales ($)':>12} {'AdFee Before':>13} {'AdFee After':>12}"
        f" {'ROAS Bef':>10} {'ROAS Aft':>10}"
        f" {'AvgCPC Bef':>11} {'AvgCPC Aft':>11} {'Clicks':>8}"
    )

    print(f"\n{'=' * len(cols)}")
    print(f"Placement Group: {pg}")
    print(cols)
    print("-" * len(cols))

    for ck in cohort_keys:
        row_match = summary[(summary["placement_group"] == pg) & (summary["cohort_key"] == ck)]
        if row_match.empty:
            print(f"{str(ck):<40} (no data)")
            continue
        row = row_match.iloc[0]

        cohort_df = df[(df["placement_group"] == pg) & (df["cohort_key"] == ck)]
        cohort_cpc_by_campaign = cohort_df.groupby("campaign_id")["cpc_dollars"].sum()
        fractions = (cohort_cpc_by_campaign / _campaign_total_cpc.reindex(cohort_cpc_by_campaign.index)).fillna(1.0)

        cohort_roas = _roas_lookup.reindex(fractions.index).dropna(subset=["total_ad_fee_usd"])
        cohort_roas = cohort_roas.join(fractions.rename("fraction"), how="left").fillna({"fraction": 1.0})

        cohort_sales  = (cohort_roas["total_attributed_sales_usd"] * cohort_roas["fraction"]).sum() / _n_roas_days if not cohort_roas.empty else float("nan")
        cohort_ad_fee = (cohort_roas["total_ad_fee_usd"]           * cohort_roas["fraction"]).sum() / _n_roas_days if not cohort_roas.empty else float("nan")
        lift_dollars  = row["total_lift_pct"] / 100 * row["segment_cpc"]
        ad_fee_after  = cohort_ad_fee + lift_dollars
        roas_before   = cohort_sales / cohort_ad_fee if cohort_ad_fee > 0 else float("nan")
        roas_after    = cohort_sales / ad_fee_after  if ad_fee_after  > 0 else float("nan")
        avg_cpc_before = row["segment_cpc"] / row["n_rows"]
        avg_cpc_after  = row["segment_cpc"] * (1 + row["total_lift_pct"] / 100) / row["n_rows"]

        print(
            f"{str(ck):<40} {cohort_sales:>12.2f} {cohort_ad_fee:>13.2f} {ad_fee_after:>12.2f}"
            f" {roas_before:>10.4f} {roas_after:>10.4f}"
            f" {avg_cpc_before:>11.4f} {avg_cpc_after:>11.4f} {row['n_rows']:>8,}"
        )


# Debug top 2 cohorts per placement group
for pg in PLACEMENT_GROUP_ORDER:
    top2 = _top_cohorts(summary, pg, n=2)
    if not top2.empty:
        debug_cohort(pg, top2["cohort_key"].tolist())
