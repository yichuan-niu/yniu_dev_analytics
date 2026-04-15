import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
plt.close("all")

# ── Constants ──────────────────────────────────────────────────────────────────
EVENT_DATE = "2026-03-25"
SAMPLE_PCT = 1   # SAMPLE (1) — consistent with base query

# ROAS thresholds to sweep: one plot per threshold
ROAS_THRESHOLDS = [0, 2, 4, 6, 8]
ROAS_SNAPSHOT_START = "2026-03-19"
ROAS_SNAPSHOT_END   = "2026-03-25"

BUDGET_DATE = "2026-03-25"

MAX_RESERVE_INCREMENT = 5.0  # tunable: upper bound of hard reserve increment to explore

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
# Restricted to winners that received a charged click (is_cpc = 1).
# event_timestamp from the click table is used to order clicks chronologically
# per campaign for budget-aware revenue lift computation.
#
# Case 1: hard_reserve is binding floor, competitive auction (finalAuctionSize > 1)
#   c1_headroom = bid - hard_reserve
#
# Case 2: GREATEST(gsp, sr) is binding floor (GREATEST(gsp, sr) > hard_reserve),
#         competitive auction (finalAuctionSize > 1)
#   c2_gap            = GREATEST(gsp, sr) - hard_reserve  (delta needed before any effect)
#   c2_headroom       = bid - GREATEST(gsp, sr)           (room above competitive floor)
#
# Case 3: single-bidder auction (finalAuctionSize = 1), hard reserve is sole floor
#   c3_headroom = bid - hard_reserve
#
# cpc_dollars is returned for every clicked winner (denominator for revenue lift %)
QUERY = """
WITH winners AS (
    SELECT
        auction_id,
        campaign_id,
        auction_bid / 100.0                                                     AS auction_bid_dollars,
        bid_price_unit_amount / 100.0                                           AS cpc_dollars,
        GET(PARSE_JSON(pricing_metadata), 'cpcGsp')::INT / 100.0               AS raw_gsp_dollars,
        GET(PARSE_JSON(pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'softReserveBeta')::FLOAT
            * GET(PARSE_JSON(pricing_metadata), 'nextBid')::INT / 100.0        AS soft_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'finalAuctionSize')::INT              AS final_auction_size
    FROM edw.ads.ads_auction_candidates_event_delta
    WHERE event_date = '{event_date}'
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(auction_id)), 100) < {sample_pct}
),
clicked AS (
    -- Only auctions that resulted in a charged click (CPC revenue actually realized).
    -- MIN(event_timestamp) gives click order for budget-aware sequential computation.
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
    cpc_dollars,
    clicked.event_timestamp,
    -- Case 1: hard reserve is the binding floor (competitive auction)
    CASE
        WHEN final_auction_size > 1
         AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
         AND cpc_dollars = hard_reserve_dollars
         AND auction_bid_dollars > hard_reserve_dollars
        THEN auction_bid_dollars - hard_reserve_dollars
        ELSE NULL
    END                                                                         AS c1_headroom,
    -- Case 2: GREATEST(gsp, soft_reserve) is the binding floor (competitive auction)
    CASE
        WHEN final_auction_size > 1
         AND auction_bid_dollars > GREATEST(raw_gsp_dollars, soft_reserve_dollars)
         AND GREATEST(raw_gsp_dollars, soft_reserve_dollars) > hard_reserve_dollars
         AND cpc_dollars = GREATEST(raw_gsp_dollars, soft_reserve_dollars)
        THEN GREATEST(raw_gsp_dollars, soft_reserve_dollars) - hard_reserve_dollars
        ELSE NULL
    END                                                                         AS c2_gap,
    CASE
        WHEN final_auction_size > 1
         AND auction_bid_dollars > GREATEST(raw_gsp_dollars, soft_reserve_dollars)
         AND GREATEST(raw_gsp_dollars, soft_reserve_dollars) > hard_reserve_dollars
         AND cpc_dollars = GREATEST(raw_gsp_dollars, soft_reserve_dollars)
        THEN auction_bid_dollars - GREATEST(raw_gsp_dollars, soft_reserve_dollars)
        ELSE NULL
    END                                                                         AS c2_headroom,
    -- Case 3: single-bidder auction, hard reserve is sole floor
    CASE
        WHEN final_auction_size = 1
         AND cpc_dollars = hard_reserve_dollars
         AND auction_bid_dollars > hard_reserve_dollars
        THEN auction_bid_dollars - hard_reserve_dollars
        ELSE NULL
    END                                                                         AS c3_headroom
FROM winners
INNER JOIN clicked ON winners.auction_id = clicked.ad_auction_id
"""


def fetch_data(event_date: str = EVENT_DATE) -> pd.DataFrame:
    query = QUERY.format(sample_pct=SAMPLE_PCT, event_date=event_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    for col in ["cpc_dollars", "c1_headroom", "c2_gap", "c2_headroom", "c3_headroom"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df


# ── ROAS query ─────────────────────────────────────────────────────────────────
ROAS_QUERY = """
SELECT
    CAMPAIGN_ID,
    ROUND(SUM(TOTAL_CX_SALES_AMOUNT_LOCAL) / 100.0, 2)      AS total_attributed_sales_usd,
    ROUND(
        SUM(TOTAL_CX_SALES_AMOUNT_LOCAL)
        / NULLIF(SUM(TOTAL_CX_AD_FEE_LOCAL), 0),
        2
    )                                                         AS roas
FROM EDW.ADS.FACT_CPG_CPC_CAMPAIGN_PERFORMANCE
WHERE SNAPSHOT_DATE BETWEEN '{start_date}' AND '{end_date}'
  AND TIMEZONE_TYPE = 'utc'
  AND REPORT_TYPE   = '[brand_cohorts] campaign'
  -- AND ADS_ENTITY_TYPE   = 'campaign'
  -- AND CAMPAIGN_VERTICAL = 'ENTERPRISE_CPG'
GROUP BY CAMPAIGN_ID
HAVING SUM(TOTAL_CX_AD_FEE_LOCAL) > 0
ORDER BY 1
"""


def fetch_roas(
    start_date: str = ROAS_SNAPSHOT_START,
    end_date: str = ROAS_SNAPSHOT_END,
) -> pd.DataFrame:
    """Return a DataFrame with columns [campaign_id, total_attributed_sales_usd, roas]."""
    query = ROAS_QUERY.format(start_date=start_date, end_date=end_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["roas"] = pd.to_numeric(df["roas"], errors="coerce")
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df


# ── Budget query ───────────────────────────────────────────────────────────────
BUDGET_QUERY = """
SELECT
    campaign_id,
    COALESCE(MAX(campaign_budget), SUM(daily_budget)) / 100 AS campaign_daily_budget_dollars
FROM PRODDB.PUBLIC.FACT_ADS_DAILY_BUDGET
WHERE date_est = '{budget_date}'
GROUP BY campaign_id
"""


def fetch_budget(budget_date: str = BUDGET_DATE) -> dict:
    """Return dict {campaign_id (str) -> daily_budget_dollars (float)}."""
    query = BUDGET_QUERY.format(budget_date=budget_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
    return {str(row[0]): float(row[1]) for row in rows if row[1] is not None}


# ── Revenue lift computation ───────────────────────────────────────────────────
def compute_revenue_lift_budget_aware(
    df: pd.DataFrame,
    budget_map: dict,
    max_delta: float = 5.0,
    high_roas_ids: set = None,
) -> pd.DataFrame:
    """
    Budget-aware revenue lift across all three cases.

    Returns DataFrame with columns: delta, c1_lift_pct, c2_lift_pct, c3_lift_pct.

    Logic:
    - total_cpc denominator = ALL clicked winners (unchanged).
    - Only campaigns present in budget_map are eligible; others are excluded entirely.
    - If high_roas_ids is provided, further restrict to that set.
    - Within each eligible campaign, clicks are processed in chronological order
      (event_timestamp). For a given delta, new_cpc = cpc_dollars + uplift.
      Once cumulative new_cpc exceeds the campaign's daily budget, no further
      uplift is attributed from that campaign.
    """
    total_cpc = float(df["cpc_dollars"].sum())

    # Filter to eligible campaigns
    eligible_mask = df["campaign_id"].isin(budget_map)
    if high_roas_ids is not None:
        eligible_mask = eligible_mask & df["campaign_id"].isin(high_roas_ids)

    work = (
        df[eligible_mask]
        .sort_values(["campaign_id", "event_timestamp"], na_position="last")
        .reset_index(drop=True)
    )

    # Pre-extract numpy arrays
    cpc    = work["cpc_dollars"].to_numpy(dtype=float)
    c1_h   = work["c1_headroom"].to_numpy(dtype=float)   # nan = not Case 1
    c2_g   = work["c2_gap"].to_numpy(dtype=float)
    c2_h   = work["c2_headroom"].to_numpy(dtype=float)
    c3_h   = work["c3_headroom"].to_numpy(dtype=float)

    notna_c1 = ~np.isnan(c1_h)
    notna_c2 = ~np.isnan(c2_g)
    notna_c3 = ~np.isnan(c3_h)

    c1_h_safe = np.nan_to_num(c1_h)
    c2_g_safe = np.nan_to_num(c2_g)
    c2_h_safe = np.nan_to_num(c2_h)
    c3_h_safe = np.nan_to_num(c3_h)

    # Campaign boundary indices (work is sorted by campaign_id)
    cmp_ids = work["campaign_id"].to_numpy()
    _, first_idx = np.unique(cmp_ids, return_index=True)
    last_idx = np.append(first_idx[1:], len(cmp_ids))
    campaign_budgets = np.array([budget_map[c] for c in cmp_ids[first_idx]], dtype=float)

    deltas = np.arange(0.0, max_delta + 0.01, 0.1)
    records = []
    for delta in deltas:
        # Per-row uplift (zero for rows that don't belong to the case)
        c1_up = np.where(notna_c1, np.minimum(c1_h_safe, delta), 0.0)
        c2_up = np.where(
            notna_c2,
            np.where(
                delta <= c2_g_safe,              0.0,
                np.where(
                    delta <= c2_g_safe + c2_h_safe,  delta - c2_g_safe,
                                                     c2_h_safe
                )
            ),
            0.0,
        )
        c3_up = np.where(notna_c3, np.minimum(c3_h_safe, delta), 0.0)

        # New CPC (with uplift) used to track budget depletion
        new_cpc = cpc + c1_up + c2_up + c3_up

        c1_total = c2_total = c3_total = 0.0
        for i, (start, end) in enumerate(zip(first_idx, last_idx)):
            budget = campaign_budgets[i]
            # funded[j] = True iff the cumulative spend through click j <= budget
            funded = np.cumsum(new_cpc[start:end]) <= budget
            c1_total += c1_up[start:end][funded].sum()
            c2_total += c2_up[start:end][funded].sum()
            c3_total += c3_up[start:end][funded].sum()

        records.append({
            "delta":       round(float(delta), 1),
            "c1_lift_pct": c1_total / total_cpc * 100,
            "c2_lift_pct": c2_total / total_cpc * 100,
            "c3_lift_pct": c3_total / total_cpc * 100,
        })

    return pd.DataFrame(records)


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_revenue_lift(
    results: pd.DataFrame,
    ax: plt.Axes,
    min_roas: float,
    event_date: str = EVENT_DATE,
) -> None:
    """
    results: DataFrame with columns delta, c1_lift_pct, c2_lift_pct, c3_lift_pct.
    Draws into the provided Axes object.
    """
    max_delta = results["delta"].max()

    ax.plot(results["delta"], results["c1_lift_pct"],
            color="steelblue", linewidth=2, label="Case 1: hard reserve is binding floor")
    ax.plot(results["delta"], results["c2_lift_pct"],
            color="darkorange", linewidth=2, label="Case 2: GSP/soft reserve is binding floor")
    ax.plot(results["delta"], results["c3_lift_pct"],
            color="seagreen", linewidth=2, label="Case 3: single-bidder, hard reserve is sole floor")

    ax.set_xlabel("Hard Reserve Increment (Δ, $)", fontsize=10)
    ax.set_ylabel("Realized Revenue Lift (%)", fontsize=10)
    ax.set_title(f"ROAS ≥ {min_roas}", fontsize=11)
    ax.set_xticks(np.arange(0, max_delta + 0.1, 0.5))
    ax.set_xlim(0, max_delta + 0.1)
    ax.set_ylim(bottom=0)
    y_max = max(
        results["c1_lift_pct"].max(),
        results["c2_lift_pct"].max(),
        results["c3_lift_pct"].max(),
    )
    ax.set_yticks(np.arange(0, y_max + 2, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)


# ── Main ───────────────────────────────────────────────────────────────────────
#%%
print("Fetching auction data from Snowflake (clicked winners only)...")
df = fetch_data()
print(f"  Total clicked winners: {len(df):,}")
print(f"  Case 1 opportunities:  {df['c1_headroom'].notna().sum():,}")
print(f"  Case 2 opportunities:  {df['c2_gap'].notna().sum():,}")
print(f"  Case 3 opportunities:  {df['c3_headroom'].notna().sum():,}")
print(f"  Total CPC ($):         {df['cpc_dollars'].sum():,.2f}")

#%%
print(f"\nFetching ROAS data ({ROAS_SNAPSHOT_START} – {ROAS_SNAPSHOT_END})...")
roas_df = fetch_roas()
print(f"  Total campaigns with ROAS data: {len(roas_df):,}")

#%%
print(f"\nFetching campaign daily budgets for {BUDGET_DATE}...")
budget_map = fetch_budget()
print(f"  Campaigns with known budget: {len(budget_map):,}")

#%%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"Realized Revenue Lift vs Hard Reserve Increment\n"
    f"(SP clicked winners, budget-aware, {EVENT_DATE})",
    fontsize=13,
)

for ax, min_roas in zip(axes.flat, ROAS_THRESHOLDS):
    high_roas_ids = set(roas_df.loc[roas_df["roas"] >= min_roas, "campaign_id"])
    eligible = df["campaign_id"].isin(budget_map) & df["campaign_id"].isin(high_roas_ids)
    print(f"\n  ROAS >= {min_roas}: {len(high_roas_ids):,} campaigns, "
          f"{eligible.sum():,} / {len(df):,} eligible clicked-winner rows")
    results = compute_revenue_lift_budget_aware(
        df,
        budget_map=budget_map,
        max_delta=MAX_RESERVE_INCREMENT,
        high_roas_ids=high_roas_ids,
    )
    plot_revenue_lift(results, ax=ax, min_roas=min_roas)

plt.tight_layout()
plt.show()
