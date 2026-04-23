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
MIN_SUBMARKET_CLICKS   = 50   # skip submarket-placement cohorts with fewer clicks than this threshold
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
# Joins with geo_intelligence.public.maindb_submarket to get submarket_name.
# NULL submarket rows (unmatched submarket_id) are labeled 'Unknown'.
QUERY = """
WITH submarkets AS (
    SELECT
        t1.id   AS submarket_id,
        t1.name AS submarket_name
    FROM geo_intelligence.public.maindb_submarket t1
),
winners AS (
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
        COALESCE(sm.submarket_name, 'Unknown')                                     AS submarket_name
    FROM edw.ads.ads_auction_candidates_event_delta acd
    LEFT JOIN submarkets sm ON acd.submarket_id = sm.submarket_id
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
    winners.submarket_name,
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
# Used to compute before/after ROAS per (submarket_name, placement group) cohort.
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


# ── Plot ───────────────────────────────────────────────────────────────────────
def _draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str, fmt: str, cmap: str = "YlGnBu") -> None:
    colormap = plt.get_cmap(cmap)
    im = ax.imshow(pivot.values, aspect="auto", cmap=colormap)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    vmin = np.nanmin(pivot.values)
    vmax = np.nanmax(pivot.values)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if not np.isnan(val):
                # use white text on dark cells, black on light cells
                norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                txt_color = "white" if norm > 0.6 else "black"
                ax.text(c, r, fmt.format(val), ha="center", va="center",
                        fontsize=13, color=txt_color)
    ax.set_title(title, fontsize=14)


def plot_heatmaps(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    pivot_lift  = summary.pivot(index="submarket_name", columns="placement_group", values="total_lift_pct")
    pivot_delta = summary.pivot(index="submarket_name", columns="placement_group", values="best_delta")

    # order rows by total revenue lift (sum across placement groups) descending, top 20 only
    row_order   = pivot_lift.sum(axis=1).sort_values(ascending=False).index[:30]
    pivot_lift  = pivot_lift.reindex(row_order)
    pivot_delta = pivot_delta.reindex(row_order)

    n_rows = max(len(pivot_lift.index), len(pivot_delta.index))
    n_cols = max(len(pivot_lift.columns), len(pivot_delta.columns))
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(n_cols * 3.5, max(5, n_rows * 0.6)),
    )
    fig.suptitle(
        f"Revenue Lift by Submarket & Placement Group\n(SP clicked winners, budget-aware, {event_date})",
        fontsize=15,
    )

    _draw_heatmap(ax1, pivot_lift,  "Total Revenue Lift (%)",          "{:.2f}%")
    _draw_heatmap(ax2, pivot_delta, "Best Hard Reserve Increment ($)", "${:.2f}")

    plt.tight_layout()
    plt.show()


def plot_roas_heatmaps(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    """
    Three side-by-side ROAS heatmaps per (submarket_name, placement group):
      Subplot 1: ROAS before hard reserve increment (current state)
      Subplot 2: ROAS after applying each cohort's best hard reserve increment
                 (sales unchanged; ad spend increases by lift_dollars)
      Subplot 3: ROAS delta (after − before)
    """
    pivot_before = summary.pivot(index="submarket_name", columns="placement_group", values="roas_before")
    pivot_after  = summary.pivot(index="submarket_name", columns="placement_group", values="roas_after")
    pivot_change = summary.pivot(index="submarket_name", columns="placement_group", values="roas_change")

    # order rows by total revenue lift (sum across placement groups) descending
    row_order    = summary.groupby("submarket_name")["total_lift_pct"].sum().sort_values(ascending=False).index[:30]
    pivot_before = pivot_before.reindex(row_order)
    pivot_after  = pivot_after.reindex(row_order)
    pivot_change = pivot_change.reindex(row_order)

    n_rows = max(len(pivot_before.index), len(pivot_after.index), len(pivot_change.index))
    n_cols = max(len(pivot_before.columns), len(pivot_after.columns), len(pivot_change.columns))
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(n_cols * 3.5 * 1.5, max(5, n_rows * 0.6)),
    )
    fig.suptitle(
        f"ROAS Before vs After Best Hard Reserve Increment\n"
        f"(SP clicked winners, budget-aware, {event_date})",
        fontsize=15,
    )

    _draw_heatmap(ax1, pivot_before, "ROAS Before", "{:.2f}")
    _draw_heatmap(ax2, pivot_after,  "ROAS After (best Δ per cohort)", "{:.2f}")
    _draw_heatmap(ax3, pivot_change, "ROAS Delta (After − Before)", "{:.2f}", cmap="RdYlGn")

    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
#%%
print("Fetching auction data from Snowflake (clicked winners only)...")
# df = fetch_data()
# df.to_pickle("../data/segment_placement_submarket_revenue_df.pkl")

df = pd.read_pickle("../data/segment_placement_submarket_revenue_df.pkl")
print(f"  Total clicked winners: {len(df):,}")
print(f"  Total CPC ($):         {df['cpc_dollars'].sum():,.2f}")

#%%
print(f"\nFetching ROAS data ({ROAS_SNAPSHOT_START} – {ROAS_SNAPSHOT_END})...")
# roas_df = fetch_roas()
# roas_df.to_pickle("../data/segment_placement_submarket_roas_df.pkl")

roas_df = pd.read_pickle("../data/segment_placement_submarket_roas_df.pkl")
print(f"  Campaigns with ROAS data: {len(roas_df):,}")

#%%
print(f"\nFetching campaign daily budgets for {EVENT_DATE}...")
# budget_df = fetch_budget()
# budget_df.to_pickle("../data/segment_placement_submarket_budget_df.pkl")

budget_df = pd.read_pickle("../data/segment_placement_submarket_budget_df.pkl")
budget_map = budget_df.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
print(f"  Campaigns with known budget: {len(budget_map):,}")

#%%
# Map placement strings to group names
df["placement_group"] = df["placement"].map(PLACEMENT_TO_GROUP).fillna("Other")

# Enumerate cohorts with enough clicks
cohorts = (
    df.groupby(["submarket_name", "placement_group"])
    .size()
    .reset_index(name="n_rows")
    .query("n_rows >= @MIN_SUBMARKET_CLICKS")
    .sort_values(["submarket_name", "placement_group"])
    .reset_index(drop=True)
)
print(f"\n  Cohorts with >= {MIN_SUBMARKET_CLICKS} clicks: {len(cohorts)}")

#%%
summary_rows = []
for _, cohort in cohorts.iterrows():
    sm = cohort["submarket_name"]
    pg = cohort["placement_group"]
    df_seg = df[(df["submarket_name"] == sm) & (df["placement_group"] == pg)]

    results, seg_cpc = compute_revenue_lift_segment(df_seg, budget_map, max_delta=MAX_RESERVE_INCREMENT)
    if results is None:
        continue

    best = results.loc[results["total_lift_pct"].idxmax()]
    summary_rows.append({
        "submarket_name":  sm,
        "placement_group": pg,
        "n_rows":          int(cohort["n_rows"]),
        "segment_cpc":     seg_cpc,
        "best_delta":      float(best["delta"]),
        "total_lift_pct":  float(best["total_lift_pct"]),
    })
    print(f"  [{sm} / {pg}]  best Δ=${best['delta']:.2f}  lift={best['total_lift_pct']:.4f}%")

#%%
summary = (
    pd.DataFrame(summary_rows)
    .sort_values("total_lift_pct", ascending=False)
    .reset_index(drop=True)
)
# Only retain cohorts with positive revenue lift for all downstream analysis and plots
# summary = summary[summary["total_lift_pct"] > 0].reset_index(drop=True)
print(f"  Cohorts with positive revenue lift: {len(summary)}")

print("\nRevenue Lift by (Submarket, Placement Group) — sorted by total lift:")
print(f"{'Submarket':<40} {'Placement Group':<15} {'Rows':>8} {'Best Δ ($)':>12} {'Total Lift (%)':>16}")
print("-" * 97)
for _, row in summary.iterrows():
    print(
        f"{row['submarket_name']:<40} {row['placement_group']:<15}"
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
plot_heatmaps(summary)

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
    sm = row["submarket_name"]
    pg = row["placement_group"]

    cohort_df = df[(df["submarket_name"] == sm) & (df["placement_group"] == pg)]
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
        "submarket_name":  sm,
        "placement_group": pg,
        "roas_before":     round(roas_before, 4),
        "roas_after":      round(roas_after,  4),
        "roas_change":     round(roas_after - roas_before, 4),
    })

roas_summary = pd.DataFrame(roas_rows)
summary = summary.merge(roas_summary, on=["submarket_name", "placement_group"], how="left")

print("\nROAS Before vs After (best Δ per cohort):")
print(f"{'Submarket':<40} {'Placement Group':<15} {'Best Δ ($)':>12} {'ROAS Before':>13} {'ROAS After':>12} {'ROAS Δ':>10}")
print("-" * 107)
for _, row in summary.sort_values("roas_before", ascending=False).iterrows():
    if pd.isna(row.get("roas_before")):
        continue
    print(
        f"{row['submarket_name']:<40} {row['placement_group']:<15}"
        f" {row['best_delta']:>12.2f} {row['roas_before']:>13.4f}"
        f" {row['roas_after']:>12.4f} {row['roas_change']:>10.4f}"
    )

#%%
plot_roas_heatmaps(summary)

#%%
def plot_cpc_heatmaps(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    """
    Three side-by-side avg-CPC heatmaps per (submarket_name, placement group):
      Subplot 1: avg CPC before hard reserve increment  (segment_cpc / n_rows)
      Subplot 2: avg CPC after best hard reserve increment
                 ((segment_cpc + lift_dollars) / n_rows)
      Subplot 3: CPC delta (after − before)
    """
    df = summary.copy()
    df["avg_cpc_before"] = df["segment_cpc"] / df["n_rows"]
    df["avg_cpc_after"]  = df["segment_cpc"] * (1 + df["total_lift_pct"] / 100) / df["n_rows"]
    df["avg_cpc_delta"]  = df["avg_cpc_after"] - df["avg_cpc_before"]

    pivot_before = df.pivot(index="submarket_name", columns="placement_group", values="avg_cpc_before")
    pivot_after  = df.pivot(index="submarket_name", columns="placement_group", values="avg_cpc_after")
    pivot_delta  = df.pivot(index="submarket_name", columns="placement_group", values="avg_cpc_delta")

    # order rows by total revenue lift (sum across placement groups) descending
    row_order    = df.groupby("submarket_name")["total_lift_pct"].sum().sort_values(ascending=False).index[:30]
    pivot_before = pivot_before.reindex(row_order)
    pivot_after  = pivot_after.reindex(row_order)
    pivot_delta  = pivot_delta.reindex(row_order)

    n_rows = max(len(p.index)   for p in [pivot_before, pivot_after, pivot_delta])
    n_cols = max(len(p.columns) for p in [pivot_before, pivot_after, pivot_delta])
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(n_cols * 3.5 * 1.5, max(5, n_rows * 0.6)),
    )
    fig.suptitle(
        f"Avg CPC Before vs After Best Hard Reserve Increment\n"
        f"(SP clicked winners, budget-aware, {event_date})",
        fontsize=15,
    )

    _draw_heatmap(ax1, pivot_before, "Avg CPC Before ($)",              "${:.3f}")
    _draw_heatmap(ax2, pivot_after,  "Avg CPC After (best Δ) ($)",      "${:.3f}")
    _draw_heatmap(ax3, pivot_delta,  "CPC Delta After − Before ($)",    "${:.3f}", cmap="RdYlGn_r")

    plt.tight_layout()
    plt.show()


plot_cpc_heatmaps(summary)

#%%
# ── Top 2 submarkets by revenue lift: lift curve per placement group ───────────
SEGMENT_MAX_DELTA = 2.0
top_submarkets = (
    summary.groupby("submarket_name")["total_lift_pct"]
    .max()
    .nlargest(2)
    .index
    .tolist()
)
segment_specs = [
    (sm, cohorts[cohorts["submarket_name"] == sm])
    for sm in top_submarkets
]

fig, axes = plt.subplots(1, len(segment_specs), figsize=(9 * len(segment_specs), 5), sharey=False)
if len(segment_specs) == 1:
    axes = [axes]
fig.suptitle(f"Revenue Lift vs Hard Reserve Increment\n(SP clicked winners, budget-aware, {EVENT_DATE})", fontsize=14)

for ax, (label, seg_cohorts) in zip(axes, segment_specs):
    for _, cohort in seg_cohorts.iterrows():
        pg = cohort["placement_group"]
        df_seg = df[(df["submarket_name"] == cohort["submarket_name"]) & (df["placement_group"] == pg)]
        results, _ = compute_revenue_lift_segment(df_seg, budget_map, max_delta=SEGMENT_MAX_DELTA)
        if results is None:
            continue
        ax.plot(results["delta"], results["total_lift_pct"], linewidth=2, label=pg)
    ax.set_xlabel("Hard Reserve Increment (Δ, $)", fontsize=12)
    ax.set_ylabel("Revenue Lift (%)", fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.set_xticks(np.arange(0, SEGMENT_MAX_DELTA + 0.1, 0.1))
    ax.set_xlim(0, SEGMENT_MAX_DELTA)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Placement Group")

plt.tight_layout()
plt.show()


#%%
# ── Debug: per-placement breakdown for given submarkets ───────────────────────
def debug_cohort(submarket_names: list) -> None:
    """
    For each submarket in submarket_names, print a per-placement breakdown:
      - Submarket sales ($)
      - Submarket ad fee before / after hard reserve increment ($)
      - ROAS before / after
      - Avg CPC before / after ($)
      - Submarket clicks (n_rows)
    """
    _roas_lookup = roas_df.set_index("campaign_id")
    cols = (
        f"{'Placement':<16} {'Sales ($)':>12} {'AdFee Before':>13} {'AdFee After':>12}"
        f" {'ROAS Bef':>10} {'ROAS Aft':>10}"
        f" {'AvgCPC Bef':>11} {'AvgCPC Aft':>11} {'Clicks':>8}"
    )

    for sm in submarket_names:
        rows = summary[summary["submarket_name"] == sm].sort_values("placement_group")
        if rows.empty:
            print(f"\nNo data for: {sm}")
            continue

        print(f"\n{'=' * len(cols)}")
        print(f"Submarket: {sm}")
        print(cols)
        print("-" * len(cols))

        for _, row in rows.iterrows():
            pg = row["placement_group"]
            cohort_df = df[(df["submarket_name"] == sm) & (df["placement_group"] == pg)]
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
                f"{pg:<16} {cohort_sales:>12.2f} {cohort_ad_fee:>13.2f} {ad_fee_after:>12.2f}"
                f" {roas_before:>10.4f} {roas_after:>10.4f}"
                f" {avg_cpc_before:>11.4f} {avg_cpc_after:>11.4f} {row['n_rows']:>8,}"
            )


debug_cohort(top_submarkets)
