import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector

plt.close("all")

# ── Constants ──────────────────────────────────────────────────────────────────
EVENT_DATE   = "2026-03-25"
SAMPLE_PCT   = 100          # campaign-level sampling
MAX_RESERVE_INCREMENT = 3.0
MIN_COHORT_ROWS = 50       # skip cohorts with too few clicked-winner rows

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
# Adds placement and L1 category (via LEFT JOIN on CPG_PRODUCT_INDEX.dd_sic).
# NULL l1_category rows (unmatched dd_sic) are labeled 'Unknown'.
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
        COALESCE(ci.L1_CATEGORY_NAME, 'Unknown')                                   AS l1_category
    FROM edw.ads.ads_auction_candidates_event_delta acd
    LEFT JOIN EDW.ADXP.CPG_PRODUCT_INDEX ci ON acd.dd_sic = ci.dd_sic
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
    winners.l1_category,
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
) -> Optional[pd.DataFrame]:
    """
    Budget-aware total revenue lift for a pre-filtered segment DataFrame.

    For each hard reserve increment delta:
      new_hr = hard_reserve + delta
      - If auction_bid < new_hr: auction is not winnable → new_cpc = 0
      - Otherwise: new_cpc = min(auction_bid, max(raw_gsp, soft_reserve, new_hr))
      change = new_cpc - cpc

    Denominator: budget-constrained baseline CPC of this segment.
    Budget is applied per campaign in chronological click order.

    Returns DataFrame with columns [delta, total_lift_pct], or None if segment is empty.
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
        return None

    competitive_floor = np.maximum(gsp, sr)  # max(raw_gsp, soft_reserve), precomputed

    deltas = np.arange(0.0, max_delta + 0.01, 0.01)
    records = []
    for delta in deltas:
        new_hr = hr + delta
        new_cpc = np.where(
            bid < new_hr,
            0.0,                                              # auction lost
            np.minimum(bid, np.maximum(competitive_floor, new_hr)),  # new charge
        )
        change = new_cpc - cpc

        total_lift = 0.0
        for i, (start, end) in enumerate(zip(first_idx, last_idx)):
            funded = np.cumsum(new_cpc[start:end]) <= campaign_budgets[i]
            total_lift += change[start:end][funded].sum()

        records.append({
            "delta":          round(float(delta), 2),
            "total_lift_pct": total_lift / total_cpc * 100,
        })

    return pd.DataFrame(records)


# ── Plot ───────────────────────────────────────────────────────────────────────
def _draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str, fmt: str) -> None:
    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
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
                        fontsize=7, color=txt_color)
    ax.set_title(title, fontsize=11)


def plot_heatmaps(summary: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    pivot_lift  = summary.pivot(index="l1_category", columns="placement_group", values="total_lift_pct")
    pivot_delta = summary.pivot(index="l1_category", columns="placement_group", values="best_delta")

    n_rows = max(len(pivot_lift.index), len(pivot_delta.index))
    n_cols = max(len(pivot_lift.columns), len(pivot_delta.columns))
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(n_cols * 3.5, max(5, n_rows * 0.6)),
    )
    fig.suptitle(
        f"Revenue Lift by L1 Category & Placement Group\n(SP clicked winners, budget-aware, {event_date})",
        fontsize=13,
    )

    _draw_heatmap(ax1, pivot_lift,  "Total Revenue Lift (%)",          "{:.2f}%")
    _draw_heatmap(ax2, pivot_delta, "Best Hard Reserve Increment ($)", "${:.2f}")

    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
#%%
print("Fetching auction data from Snowflake (clicked winners only)...")
# df = fetch_data()
# df.to_pickle("data/segment_revenue_df.pkl")
df = pd.read_pickle("data/segment_revenue_df.pkl")

print(f"  Total clicked winners: {len(df):,}")
print(f"  Total CPC ($):         {df['cpc_dollars'].sum():,.2f}")

#%%
print(f"\nFetching campaign daily budgets for {EVENT_DATE}...")
# budget_df = fetch_budget()
# budget_df.to_pickle("data/segment_revenue_budget_df.pkl")
budget_df = pd.read_pickle("data/segment_revenue_budget_df.pkl")
budget_map = budget_df.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
print(f"  Campaigns with known budget: {len(budget_map):,}")

#%%
# Map placement strings to group names
df["placement_group"] = df["placement"].map(PLACEMENT_TO_GROUP).fillna("Other")

# Enumerate cohorts with enough data
cohorts = (
    df.groupby(["l1_category", "placement_group"])
    .size()
    .reset_index(name="n_rows")
    .query("n_rows >= @MIN_COHORT_ROWS")
    .sort_values(["l1_category", "placement_group"])
    .reset_index(drop=True)
)
print(f"\n  Cohorts with >= {MIN_COHORT_ROWS} rows: {len(cohorts)}")

#%%
summary_rows = []
for _, cohort in cohorts.iterrows():
    l1 = cohort["l1_category"]
    pg = cohort["placement_group"]
    df_seg = df[(df["l1_category"] == l1) & (df["placement_group"] == pg)]

    results = compute_revenue_lift_segment(df_seg, budget_map, max_delta=MAX_RESERVE_INCREMENT)
    if results is None:
        continue

    best = results.loc[results["total_lift_pct"].idxmax()]
    summary_rows.append({
        "l1_category":    l1,
        "placement_group": pg,
        "n_rows":          int(cohort["n_rows"]),
        "best_delta":      float(best["delta"]),
        "total_lift_pct":  float(best["total_lift_pct"]),
    })
    print(f"  [{l1} / {pg}]  best Δ=${best['delta']:.2f}  lift={best['total_lift_pct']:.4f}%")

#%%
summary = pd.DataFrame(summary_rows).sort_values("total_lift_pct", ascending=False).reset_index(drop=True)

print("\nRevenue Lift by (L1 Category, Placement Group) — sorted by total lift:")
print(f"{'L1 Category':<35} {'Placement Group':<15} {'Rows':>8} {'Best Δ ($)':>12} {'Total Lift (%)':>16}")
print("-" * 92)
for _, row in summary.iterrows():
    print(
        f"{row['l1_category']:<35} {row['placement_group']:<15}"
        f" {row['n_rows']:>8,} {row['best_delta']:>12.2f} {row['total_lift_pct']:>16.4f}"
    )

#%%
plot_heatmaps(summary)