import os
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
plt.close("all")

# ── Constants ──────────────────────────────────────────────────────────────────
EVENT_DATE = "2026-03-25"
SAMPLE_PCT = 1

# ── Placement groups ───────────────────────────────────────────────────────────
GROUPS = {
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
    "DoubleDash Store": [
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_COLLECTION",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_CATEGORY_L1",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_CATEGORY_L2",
        "PLACEMENT_TYPE_SPONSORED_PRODUCTS_DOUBLE_DASH_STORE_SEARCH",
    ],
}

ALL_PLACEMENTS = [p for placements in GROUPS.values() for p in placements]

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
# Fetches all auction candidates (winners + losers) across the target placements,
# with a flag for whether the auction resulted in a charged click.
# auction_rank = 0 → winner; > 0 → loser.
QUERY = """
WITH candidates AS (
    SELECT
        auction_id,
        auction_bid / 100.0                         AS auction_bid_dollars,
        auction_rank,
        placement
    FROM edw.ads.ads_auction_candidates_event_delta
    WHERE event_date = '{event_date}'
      AND CURRENCY_ISO_TYPE in ('USD')
      AND placement IN ({placements})
      AND pricing_metadata IS NOT NULL
      AND auction_rank < 5
      AND MOD(ABS(HASH(auction_id)), 100) < {sample_pct}
),
clicked AS (
    SELECT DISTINCT ad_auction_id
    FROM proddb.public.fact_item_card_click_dedup
    WHERE event_date = '{event_date}'
      AND is_sponsored = 1
      AND is_cpc = 1
      AND ad_auction_id IS NOT NULL
)
SELECT
    candidates.auction_bid_dollars,
    candidates.auction_rank,
    candidates.placement,
    (clicked.ad_auction_id IS NOT NULL)::BOOLEAN    AS is_clicked
FROM candidates
LEFT JOIN clicked ON candidates.auction_id = clicked.ad_auction_id
"""


def fetch_data(event_date: str = EVENT_DATE) -> pd.DataFrame:
    placements_sql = ", ".join(f"'{p}'" for p in ALL_PLACEMENTS)
    query = QUERY.format(
        event_date=event_date,
        placements=placements_sql,
        sample_pct=SAMPLE_PCT,
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["auction_bid_dollars"] = pd.to_numeric(df["auction_bid_dollars"], errors="coerce")
    df["auction_rank"] = pd.to_numeric(df["auction_rank"], errors="coerce")
    df["is_clicked"] = df["is_clicked"].astype(bool)
    return df


# ── Subset selectors (4 plot types) ───────────────────────────────────────────
PLOT_TITLES = [
    "Plot 1: All Winners",
    "Plot 2: All Candidates (Winners + Losers)",
    "Plot 3: Winners from Clicked Auctions",
    "Plot 4: All Candidates from Clicked Auctions",
]

def get_subset(df: pd.DataFrame, plot_idx: int) -> pd.Series:
    if plot_idx == 0:   # all winners
        return df.loc[df["auction_rank"] == 0, "auction_bid_dollars"]
    elif plot_idx == 1: # all candidates
        return df["auction_bid_dollars"]
    elif plot_idx == 2: # winners from clicked auctions
        return df.loc[(df["auction_rank"] == 0) & df["is_clicked"], "auction_bid_dollars"]
    else:               # all candidates from clicked auctions
        return df.loc[df["is_clicked"], "auction_bid_dollars"]


# ── Plot helper ────────────────────────────────────────────────────────────────
def plot_histogram(ax: plt.Axes, bids: pd.Series, title: str) -> None:
    import numpy as np
    bids = bids.dropna()
    cap = bids.quantile(0.99)          # clip top 1% outliers for readability
    bids_clipped = bids[bids <= cap]
    ax.hist(bids_clipped, bins=60, color="steelblue", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("Auction Bid ($)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

    # Dense ticks below $1 (every $0.10) to show the hard reserve floor clearly;
    # coarser ticks above $1.
    sub_dollar = np.arange(0, 1.0, 0.20)
    above_dollar = np.arange(1.0, cap + 0.5, max(0.5, round((cap - 1.0) / 8, 1)))
    xticks = np.unique(np.concatenate([sub_dollar, above_dollar]))
    ax.set_xticks(xticks[xticks <= cap])
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.text(0.97, 0.96, f"n={len(bids):,}\np99 cap=${cap:.2f}",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            color="dimgray")


# ── Main ───────────────────────────────────────────────────────────────────────
#%%
print("Fetching data from Snowflake...")
df = fetch_data()
print(f"  Total rows: {len(df):,}")
for name, placements in GROUPS.items():
    n = df["placement"].isin(placements).sum()
    print(f"  {name}: {n:,} rows")

#%%
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
fig.suptitle(
    f"Auction Bid Histogram by Placement Group  |  SP  |  {EVENT_DATE}",
    fontsize=13, fontweight="bold",
)

for row_idx, (group_name, placements) in enumerate(GROUPS.items()):
    group_df = df[df["placement"].isin(placements)]
    for col_idx in range(4):
        ax = axes[row_idx, col_idx]
        bids = get_subset(group_df, col_idx)
        plot_histogram(ax, bids, PLOT_TITLES[col_idx])
        if col_idx == 0:
            ax.set_ylabel(f"[{group_name}]\nCount", fontsize=8)

plt.tight_layout()
plt.show()
