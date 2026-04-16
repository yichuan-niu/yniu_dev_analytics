import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector

# ── Constants ──────────────────────────────────────────────────────────────────
EVENT_DATE = "2026-03-25"
SAMPLE_PCT = 1   # SAMPLE (1) — consistent with base query


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
# Single CTE covers all three cases. NULLs indicate a row does not belong to that case.
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
# cpc_dollars is returned for every winner (denominator for revenue lift %)
QUERY = """
WITH winners AS (
    SELECT
        auction_bid / 100.0                                                     AS auction_bid_dollars,
        bid_price_unit_amount / 100.0                                           AS cpc_dollars,
        GET(PARSE_JSON(pricing_metadata), 'cpcGsp')::INT / 100.0               AS raw_gsp_dollars,
        GET(PARSE_JSON(pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'softReserveBeta')::FLOAT
            * GET(PARSE_JSON(pricing_metadata), 'nextBid')::INT / 100.0        AS soft_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'finalAuctionSize')::INT              AS final_auction_size
    FROM edw.ads.ads_auction_candidates_event_delta
    WHERE event_date = '{event_date}'
      AND CURRENCY_ISO_TYPE in ('USD')
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(auction_id)), 100) < {sample_pct}
)
SELECT
    cpc_dollars,
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
    return df


# ── Revenue lift computation ───────────────────────────────────────────────────
def compute_revenue_lift_case1(df: pd.DataFrame, max_delta: float = 1.5) -> pd.DataFrame:
    """
    Case 1: hard reserve is already the binding floor.
      Per-auction uplift = min(delta, c1_headroom)
      revenue_lift_pct   = sum(min(delta, c1_headroom)) / total_cpc * 100
    """
    total_cpc = float(df["cpc_dollars"].sum())
    headroom = df["c1_headroom"].dropna().astype(float).values

    deltas = np.arange(0.0, max_delta + 0.1, 0.1)
    revenue_lift_pct = [
        np.minimum(headroom, delta).sum() / total_cpc * 100
        for delta in deltas
    ]
    return pd.DataFrame({"delta": deltas, "revenue_lift_pct": revenue_lift_pct})


def compute_revenue_lift_case2(df: pd.DataFrame, max_delta: float = 1.5) -> pd.DataFrame:
    """
    Case 2: GREATEST(gsp, soft_reserve) is the binding floor above hard_reserve.
    For a given delta (hard reserve increment), per-auction uplift follows 3 conditions:
      1. new_hard_reserve <= gsp_floor  (delta <= gap):
             uplift = 0  (hard reserve still below competitive floor, no effect)
      2. gsp_floor < new_hard_reserve <= auction_bid  (gap < delta <= gap + c2_headroom):
             uplift = new_hard_reserve - gsp_floor = delta - gap  (charge by hard reserve)
      3. new_hard_reserve > auction_bid  (delta > gap + c2_headroom):
             uplift = auction_bid - gsp_floor = c2_headroom  (capped at winner's bid)
    where:
      gap         = gsp_floor - hard_reserve  (increment needed before any effect)
      c2_headroom = auction_bid - gsp_floor   (max possible uplift per auction)
    """
    total_cpc = float(df["cpc_dollars"].sum())
    case2 = df[df["c2_gap"].notna()].copy()
    gap = case2["c2_gap"].astype(float).values
    headroom = case2["c2_headroom"].astype(float).values

    deltas = np.arange(0.0, max_delta + 0.1, 0.1)
    revenue_lift_pct = [
        np.where(
            delta <= gap,                 headroom * 0,       # condition 1: no effect
            np.where(
                delta <= gap + headroom,  delta - gap,        # condition 2: charge by hard reserve
                                          headroom            # condition 3: capped at bid
            )
        ).sum() / total_cpc * 100
        for delta in deltas
    ]
    return pd.DataFrame({"delta": deltas, "revenue_lift_pct": revenue_lift_pct})


def compute_revenue_lift_case3(df: pd.DataFrame, max_delta: float = 1.5) -> pd.DataFrame:
    """
    Case 3: single-bidder auction, hard reserve is the sole price floor.
      Per-auction uplift = min(delta, c3_headroom)
      revenue_lift_pct   = sum(min(delta, c3_headroom)) / total_cpc * 100
    """
    total_cpc = float(df["cpc_dollars"].sum())
    headroom = df["c3_headroom"].dropna().astype(float).values

    deltas = np.arange(0.0, max_delta + 0.1, 0.1)
    revenue_lift_pct = [
        np.minimum(headroom, delta).sum() / total_cpc * 100
        for delta in deltas
    ]
    return pd.DataFrame({"delta": deltas, "revenue_lift_pct": revenue_lift_pct})


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_revenue_lift(
    results_c1: pd.DataFrame,
    results_c2: pd.DataFrame,
    results_c3: pd.DataFrame,
    event_date: str = EVENT_DATE,
) -> None:
    max_delta = results_c1["delta"].max()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(results_c1["delta"], results_c1["revenue_lift_pct"],
            color="steelblue", linewidth=3, label="Case 1: hard reserve is binding floor")
    ax.plot(results_c2["delta"], results_c2["revenue_lift_pct"],
            color="darkorange", linewidth=3, label="Case 2: GSP/soft reserve is binding floor")
    ax.plot(results_c3["delta"], results_c3["revenue_lift_pct"],
            color="seagreen", linewidth=3, label="Case 3: single-bidder, hard reserve is sole floor")

    ax.set_xlabel("Hard Reserve Increment (Δ, $)", fontsize=12)
    ax.set_ylabel("Revenue Lift (%)", fontsize=12)
    ax.set_title(f"Revenue Lift vs Hard Reserve Increment\n(SP winners, {event_date})", fontsize=13)
    ax.set_xticks(np.arange(0, max_delta + 0.1, 0.2))
    ax.set_xlim(0, max_delta + 0.1)
    ax.set_ylim(bottom=0)
    y_max = max(results_c1["revenue_lift_pct"].max(),
                results_c2["revenue_lift_pct"].max(),
                results_c3["revenue_lift_pct"].max())
    ax.set_yticks(np.arange(0, y_max + 2, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.legend()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
#%%
print("Fetching data from Snowflake...")
df = fetch_data()
print(f"  Total winners:        {len(df):,}")
print(f"  Case 1 opportunities: {df['c1_headroom'].notna().sum():,}")
print(f"  Case 2 opportunities: {df['c2_gap'].notna().sum():,}")
print(f"  Case 3 opportunities: {df['c3_headroom'].notna().sum():,}")
print(f"  Total CPC ($):        {df['cpc_dollars'].sum():,.2f}")

#%%
max_delta = 5.0  # tunable: upper bound of hard reserve increment to explore
results_c1 = compute_revenue_lift_case1(df, max_delta=max_delta)
results_c2 = compute_revenue_lift_case2(df, max_delta=max_delta)
results_c3 = compute_revenue_lift_case3(df, max_delta=max_delta)
plot_revenue_lift(results_c1, results_c2, results_c3)
