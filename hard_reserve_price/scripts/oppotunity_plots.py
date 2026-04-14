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
# Pull two things from a single CTE:
#   1. headroom (bid - hard_reserve) for every opportunity auction
#   2. total CPC across all winners (denominator for revenue lift %)
QUERY = """
WITH winners AS (
    SELECT
        auction_bid / 100.0                                                     AS auction_bid_dollars,
        bid_price_unit_amount / 100.0                                           AS cpc_dollars,
        GET(PARSE_JSON(pricing_metadata), 'cpcGsp')::INT / 100.0               AS raw_gsp_dollars,
        GET(PARSE_JSON(pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'softReserveBeta')::FLOAT
            * GET(PARSE_JSON(pricing_metadata), 'nextBid')::INT / 100.0        AS soft_reserve_dollars
    FROM edw.ads.ads_auction_candidates_event_delta SAMPLE ({sample_pct})
    WHERE event_date = '{event_date}'
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND GET(PARSE_JSON(pricing_metadata), 'finalAuctionSize')::INT > 1
)
SELECT
    -- headroom: NULL for non-opportunity auctions, positive value for opportunity ones
    CASE
        WHEN hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
         AND cpc_dollars = hard_reserve_dollars
         AND auction_bid_dollars > hard_reserve_dollars
        THEN auction_bid_dollars - hard_reserve_dollars
        ELSE NULL
    END                                                                         AS headroom,
    cpc_dollars                                                                 AS cpc_dollars
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
    df["headroom"] = pd.to_numeric(df["headroom"], errors="coerce")
    df["cpc_dollars"] = pd.to_numeric(df["cpc_dollars"], errors="coerce")
    return df


# ── Revenue lift computation ───────────────────────────────────────────────────
def compute_revenue_lift(df: pd.DataFrame, max_delta: float = 1.5) -> pd.DataFrame:
    """
    For each delta in [0, max_delta], compute revenue lift %:
      - If headroom >= delta: winner pays hard_reserve + delta, uplift = delta
      - If headroom < delta:  winner is capped at their bid,    uplift = headroom
      - Per-auction uplift = min(delta, headroom)
      - revenue_lift_pct = sum(min(delta, headroom)) / total_cpc * 100
    """
    total_cpc = float(df["cpc_dollars"].sum())
    headroom = df["headroom"].dropna().astype(float).values

    deltas = np.arange(0.0, max_delta + 0.01, 0.01)
    revenue_lift_pct = [
        np.minimum(headroom, delta).sum() / total_cpc * 100
        for delta in deltas
    ]

    return pd.DataFrame({"delta": deltas, "revenue_lift_pct": revenue_lift_pct})


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_revenue_lift(results: pd.DataFrame, event_date: str = EVENT_DATE) -> None:
    plt.close("all")
    fig, ax = plt.subplots()
    max_delta = results["delta"].max()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["delta"], results["revenue_lift_pct"], color="steelblue", linewidth=2)
    ax.set_xlabel("Hard Reserve Increment (Δ, $)", fontsize=12)
    ax.set_ylabel("Revenue Lift (%)", fontsize=12)
    ax.set_title(f"Revenue Lift vs Hard Reserve Increment\n(SP winners, {event_date})", fontsize=13)
    ax.set_xticks(np.arange(0, max_delta + 0.1, 0.1))
    ax.set_xlim(0, max_delta + 0.1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

#%%
print("Fetching data from Snowflake...")
df = fetch_data()
print(f"  Total winners:       {len(df):,}")
print(f"  Opportunity winners: {df['headroom'].notna().sum():,}")
print(f"  Total CPC ($):       {df['cpc_dollars'].sum():,.2f}")
#%%
max_delta = 1.5  # tunable: upper bound of hard reserve increment to explore
results = compute_revenue_lift(df, max_delta=max_delta)
plot_revenue_lift(results)
