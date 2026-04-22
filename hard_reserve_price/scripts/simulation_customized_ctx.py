import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from scipy.optimize import minimize, brentq
from scipy.stats import gamma as gamma_dist, lognorm

plt.close("all")

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_START_DATE    = "2026-03-25"   # training window start (inclusive)
TRAIN_END_DATE      = "2026-03-31"   # training window end (inclusive)
EVAL_START_DATE     = "2026-04-01"   # evaluation window start (inclusive)
EVAL_END_DATE       = "2026-04-03"   # evaluation window end (inclusive)
TRAIN_SAMPLE_PCT    = 5              # auction-level sampling for training (MOD HASH < TRAIN_SAMPLE_PCT)
EVAL_SAMPLE_PCT     = 50             # campaign-level sampling for eval (MOD HASH < SAMPLE_PCT)
MAX_RANK            = 5              # use auction_rank < MAX_RANK for training bids
MIN_COHORT_BIDS     = 100            # min bid rows per cohort to fit a distribution
DIST_TYPE           = "gamma"        # "gamma" or "lognormal"
SELLER_VALUE        = 0.0            # Myerson seller valuation (v_0), usually 0

# Default hard reserve / floor price per placement group (USD).
# Used as truncation point in truncated MLE and as the minimum allowed reserve.
FLOOR_PRICES = {
    "Search":     0.60,
    "Category":   0.40,
    "Collection": 0.30,
    "DoubleDash": 0.80,
}

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

# Secondary cohort dimension per placement group (matches segment_placement_ctx.py).
COHORT_DIM = {
    "Search":     "normalized_query",
    "Category":   "l1_category_id",
    "Collection": "collection_id",
    "DoubleDash": "hour_bucket",
}
PLACEMENT_GROUP_ORDER = ["Search", "Category", "Collection", "DoubleDash"]

# ── Snowflake connection ──────────────────────────────────────────────────────
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


# ── Training SQL ──────────────────────────────────────────────────────────────
# Pulls all auction candidates (rank < MAX_RANK) from auctions that had a click,
# over the N-day training window.  Floor filtering is done in Python per group.
# event_hour from the auction table is used as hour_bucket for DoubleDash.
TRAINING_QUERY = """
WITH clicked_auctions AS (
    SELECT DISTINCT ad_auction_id
    FROM proddb.public.fact_item_card_click_dedup
    WHERE event_date BETWEEN '{train_start_date}' AND '{train_end_date}'
      AND is_sponsored = 1
      AND is_cpc = 1
      AND ad_auction_id IS NOT NULL
      AND campaign_id IS NOT NULL
)
SELECT
    acd.placement,
    acd.auction_bid / 100.0                                             AS auction_bid_dollars,
    GET(PARSE_JSON(acd.pricing_metadata), 'hardReserve')::INT / 100.0  AS hard_reserve_dollars,
    COALESCE(acd.normalized_query, 'Unknown')                          AS normalized_query,
    COALESCE(acd.l1_category_id::VARCHAR, 'Unknown')                   AS l1_category_id,
    COALESCE(acd.collection_id, 'Unknown')                             AS collection_id,
    acd.event_hour                                                      AS hour_bucket
FROM edw.ads.ads_auction_candidates_event_delta acd
INNER JOIN clicked_auctions ON acd.auction_id = clicked_auctions.ad_auction_id
WHERE acd.event_date BETWEEN '{train_start_date}' AND '{train_end_date}'
  AND acd.CURRENCY_ISO_TYPE IN ('USD')
  AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
  AND acd.auction_rank < {max_rank}
  AND acd.pricing_metadata IS NOT NULL
  AND MOD(ABS(HASH(acd.auction_id)), 100) < {train_sample_pct}
"""

# ── Evaluation SQL ────────────────────────────────────────────────────────────
# Clicked winners (rank=0) over the eval date range with all pricing and cohort fields.
# Identical structure to segment_placement_ctx.py QUERY.
EVAL_QUERY = """
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
        COALESCE(acd.l1_category_id::VARCHAR, 'Unknown')                           AS l1_category_id,
        COALESCE(acd.collection_id, 'Unknown')                                     AS collection_id,
        acd.event_hour                                                              AS hour_bucket
    FROM edw.ads.ads_auction_candidates_event_delta acd
    WHERE acd.event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
      AND acd.CURRENCY_ISO_TYPE IN ('USD')
      AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
      AND acd.auction_rank = 0
      AND acd.pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(acd.campaign_id)), 100) < {eval_sample_pct}
),
clicked AS (
    SELECT
        ad_auction_id,
        MIN(event_timestamp) AS event_timestamp
    FROM proddb.public.fact_item_card_click_dedup
    WHERE event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
      AND is_sponsored = 1
      AND is_cpc = 1
      AND ad_auction_id IS NOT NULL
      AND campaign_id IS NOT NULL
    GROUP BY ad_auction_id
)
SELECT
    winners.auction_id,
    winners.campaign_id,
    winners.placement,
    winners.normalized_query,
    winners.l1_category_id,
    winners.collection_id,
    winners.hour_bucket,
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

SALES_QUERY = """
SELECT
    AD_AUCTION_ID                                       AS auction_id,
    SUM(PRICE_UNIT_AMOUNT * QUANTITY_RECEIVED) / 100.0 AS attributed_sales_usd
FROM proddb.public.FACT_ITEM_ORDER_ATTRIBUTION
WHERE event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
GROUP BY AD_AUCTION_ID
"""


# ── Data fetchers ─────────────────────────────────────────────────────────────
def fetch_train_data(
    train_start_date: str = TRAIN_START_DATE,
    train_end_date: str = TRAIN_END_DATE,
    train_sample_pct: int = TRAIN_SAMPLE_PCT,
    max_rank: int = MAX_RANK,
) -> pd.DataFrame:
    query = TRAINING_QUERY.format(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        max_rank=max_rank,
        train_sample_pct=train_sample_pct,
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["auction_bid_dollars"] = pd.to_numeric(df["auction_bid_dollars"], errors="coerce")
    df["hard_reserve_dollars"] = pd.to_numeric(df["hard_reserve_dollars"], errors="coerce")
    df["hour_bucket"] = pd.to_numeric(df["hour_bucket"], errors="coerce").astype("Int64")
    df["campaign_id"] = df.get("campaign_id", pd.Series(dtype=str)).astype(str)
    return df


def fetch_eval_data(
    eval_start_date: str = EVAL_START_DATE,
    eval_end_date: str = EVAL_END_DATE,
    eval_sample_pct: int = EVAL_SAMPLE_PCT,
) -> pd.DataFrame:
    query = EVAL_QUERY.format(
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
        eval_sample_pct=eval_sample_pct,
    )
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
    df["hour_bucket"] = pd.to_numeric(df["hour_bucket"], errors="coerce").astype("Int64")
    df["auction_id"] = df["auction_id"].astype(str)
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df


def fetch_budget(budget_date: str = EVAL_END_DATE) -> pd.DataFrame:
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


def fetch_sales(
    eval_start_date: str = EVAL_START_DATE,
    eval_end_date: str = EVAL_END_DATE,
) -> pd.DataFrame:
    """Fetch per-auction attributed sales from FACT_ITEM_ORDER_ATTRIBUTION."""
    query = SALES_QUERY.format(
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["auction_id"] = df["auction_id"].astype(str)
    df["attributed_sales_usd"] = pd.to_numeric(df["attributed_sales_usd"], errors="coerce")
    return df.dropna(subset=["attributed_sales_usd"])


# ── Distribution fitting (truncated MLE) ─────────────────────────────────────
def _gamma_nll(params, bids: np.ndarray, floor: float) -> float:
    """Negative log-likelihood for Gamma truncated from below at floor."""
    alpha, theta = params
    d = gamma_dist(a=alpha, scale=theta)
    surv = 1.0 - d.cdf(floor)
    if surv <= 0:
        return 1e18
    return -(np.sum(d.logpdf(bids)) - len(bids) * np.log(surv))


def fit_gamma_truncated(bids: np.ndarray, floor: float):
    """Fit a Gamma distribution to bids truncated from below at floor via L-BFGS-B."""
    m = bids.mean()
    v = bids.var() if bids.var() > 0 else m
    theta0 = v / m
    alpha0 = m / theta0
    result = minimize(
        _gamma_nll,
        x0=[alpha0, theta0],
        args=(bids, floor),
        method="L-BFGS-B",
        bounds=[(1e-3, None), (1e-6, None)],
    )
    alpha, theta = result.x
    return gamma_dist(a=alpha, scale=theta)


def _lognorm_nll(params, bids: np.ndarray, floor: float) -> float:
    """Negative log-likelihood for Lognormal truncated from below at floor."""
    mu, sigma = params
    d = lognorm(s=sigma, scale=np.exp(mu))
    surv = 1.0 - d.cdf(floor)
    if surv <= 0:
        return 1e18
    return -(np.sum(d.logpdf(bids)) - len(bids) * np.log(surv))


def fit_lognormal_truncated(bids: np.ndarray, floor: float):
    """Fit a Lognormal distribution to bids truncated from below at floor via L-BFGS-B."""
    log_bids = np.log(bids)
    result = minimize(
        _lognorm_nll,
        x0=[log_bids.mean(), log_bids.std() if log_bids.std() > 0 else 0.5],
        args=(bids, floor),
        method="L-BFGS-B",
        bounds=[(None, None), (1e-6, None)],
    )
    mu, sigma = result.x
    return lognorm(s=sigma, scale=np.exp(mu))


def fit_distribution(bids: np.ndarray, floor: float, dist_type: str = DIST_TYPE):
    """Dispatcher: returns a fitted scipy frozen distribution."""
    if dist_type == "lognormal":
        return fit_lognormal_truncated(bids, floor)
    return fit_gamma_truncated(bids, floor)


# ── Myerson optimal reserve ───────────────────────────────────────────────────
def virtual_valuation(v: float, dist) -> float:
    """Myerson virtual valuation: ψ(v) = v - (1 - F(v)) / f(v)"""
    pdf_v = dist.pdf(v)
    if pdf_v <= 0:
        return float("inf")
    return v - (1 - dist.cdf(v)) / pdf_v


def myerson_optimal_reserve(
    dist,
    floor: float,
    seller_value: float = SELLER_VALUE,
    hi: float = 50.0,
):
    """
    Find r* where ψ(r*) = seller_value using Brent's bisection method.

    Returns the optimal reserve price if r* > floor, else None
    (meaning the theoretically optimal reserve is at or below the current floor,
    so no change is needed).
    """
    vv = lambda v: virtual_valuation(v, dist) - seller_value
    # If virtual valuation at just above the floor is already >= 0,
    # the optimal reserve <= floor — no improvement possible.
    try:
        if vv(floor + 1e-6) >= 0:
            return None
        r_star = brentq(vv, 1e-6, hi, xtol=1e-4)
    except ValueError:
        return None
    return r_star if r_star > floor else None


# ── Training: fit per-cohort distribution and solve Myerson ──────────────────
def _add_cohort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add placement_group and cohort_key columns (mirrors segment_placement_ctx.py)."""
    df = df.copy()
    df["placement_group"] = df["placement"].map(PLACEMENT_TO_GROUP).fillna("Other")
    df["hour_bucket"] = df["hour_bucket"].astype(str)
    conditions = [df["placement_group"] == pg for pg in PLACEMENT_GROUP_ORDER]
    choices    = [df[COHORT_DIM[pg]] for pg in PLACEMENT_GROUP_ORDER]
    df["cohort_key"] = np.select(conditions, choices, default="Other")
    return df


def train_optimal_reserves(
    train_df: pd.DataFrame,
    min_cohort_bids: int = MIN_COHORT_BIDS,
    dist_type: str = DIST_TYPE,
) -> dict:
    """
    For each (placement_group, cohort_key) cohort with enough bid data:
      1. Filter bids > floor (removes floor-clamped auto-bid values).
      2. Fit Gamma (or lognormal) via truncated MLE.
      3. Solve Myerson's virtual valuation = 0 with Brent's method.
      4. Keep r* only if r* > floor.

    Returns dict[(placement_group, cohort_key)] = r_star.
    """
    df = _add_cohort_columns(train_df)
    df = df[df["placement_group"].isin(PLACEMENT_GROUP_ORDER)].copy()

    optimal_hr: dict = {}
    skipped_small = skipped_solve = 0

    cohorts = (
        df.groupby(["placement_group", "cohort_key"])
        .size()
        .reset_index(name="n_total")
    )

    for _, cohort in cohorts.iterrows():
        pg = cohort["placement_group"]
        ck = cohort["cohort_key"]
        floor = FLOOR_PRICES[pg]

        mask = (df["placement_group"] == pg) & (df["cohort_key"] == ck)
        bids = df.loc[mask, "auction_bid_dollars"].dropna().to_numpy(dtype=float)
        bids = bids[bids > floor]  # drop floor-clamped values

        if len(bids) < min_cohort_bids:
            skipped_small += 1
            continue

        try:
            dist = fit_distribution(bids, floor, dist_type)
            r_star = myerson_optimal_reserve(dist, floor)
        except Exception:
            skipped_solve += 1
            continue

        if r_star is None:
            skipped_solve += 1
            continue

        optimal_hr[(pg, ck)] = r_star
        print(f"  [{pg} / {ck}]  floor=${floor:.2f}  r*=${r_star:.4f}  n={len(bids):,}")

    print(f"\n  Cohorts solved: {len(optimal_hr)}")
    print(f"  Skipped (too few bids): {skipped_small}")
    print(f"  Skipped (no root / fit error): {skipped_solve}")
    return optimal_hr


# ── Evaluation: budget-aware auction replay ───────────────────────────────────
def _evaluate_cohort(
    df_seg: pd.DataFrame,
    budget_map: dict,
    new_hr: float,
) -> tuple[float, float, int]:
    """
    Budget-aware replay for a single cohort at a fixed new hard reserve new_hr.

    Returns (ad_fee_before, ad_fee_after, n_clicks).
    Mirrors compute_revenue_lift_segment in segment_placement_ctx.py.
    """
    work = (
        df_seg
        .sort_values(["campaign_id", "event_timestamp"], na_position="last")
        .reset_index(drop=True)
    )
    bid = work["auction_bid_dollars"].to_numpy(dtype=float)
    gsp = work["raw_gsp_dollars"].to_numpy(dtype=float)
    sr  = work["soft_reserve_dollars"].to_numpy(dtype=float)
    hr  = work["hard_reserve_dollars"].to_numpy(dtype=float)
    cmp_ids = work["campaign_id"].to_numpy()

    _, first_idx = np.unique(cmp_ids, return_index=True)
    last_idx = np.append(first_idx[1:], len(cmp_ids))
    budgets = np.array(
        [budget_map.get(c, float("inf")) for c in cmp_ids[first_idx]], dtype=float
    )

    competitive_floor = np.maximum(gsp, sr)

    # Baseline CPC anchored to formula (avoids data edge-case drift)
    cpc_baseline = np.where(
        bid < hr,
        0.0,
        np.minimum(bid, np.maximum(competitive_floor, hr)),
    )
    new_cpc = np.where(
        bid < new_hr,
        0.0,
        np.minimum(bid, np.maximum(competitive_floor, new_hr)),
    )

    ad_fee_before = ad_fee_after = 0.0
    for i, (start, end) in enumerate(zip(first_idx, last_idx)):
        funded = np.cumsum(new_cpc[start:end]) <= budgets[i]
        # "before" is also budget-gated on new_cpc spend order (consistent denominator)
        ad_fee_before += cpc_baseline[start:end][funded].sum()
        ad_fee_after  += new_cpc[start:end][funded].sum()

    return float(ad_fee_before), float(ad_fee_after), len(work)


def evaluate_all_cohorts(
    eval_df: pd.DataFrame,
    budget_map: dict,
    optimal_hr_map: dict,
) -> pd.DataFrame:
    """
    Replay evaluation data using the Myerson-optimal reserve per cohort.

    eval_df must already have 'placement_group' and 'cohort_key' columns
    (set by the caller via _add_cohort_columns or equivalent).

    For each (placement_group, cohort_key) that appears in eval_df:
      - Look up r_star from optimal_hr_map.
      - Only apply r_star if r_star > FLOOR_PRICES[pg]; otherwise no change.

    Returns a summary DataFrame per cohort with before/after metrics.
    """
    df = eval_df[eval_df["placement_group"].isin(PLACEMENT_GROUP_ORDER)].copy()

    cohorts = (
        df.groupby(["placement_group", "cohort_key"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["placement_group", "cohort_key"])
        .reset_index(drop=True)
    )

    rows = []
    for _, cohort in cohorts.iterrows():
        pg = cohort["placement_group"]
        ck = cohort["cohort_key"]
        floor = FLOOR_PRICES[pg]

        r_star = optimal_hr_map.get((pg, ck))
        new_hr = r_star if (r_star is not None and r_star > floor) else floor

        df_seg = df[(df["placement_group"] == pg) & (df["cohort_key"] == ck)]
        before, after, n = _evaluate_cohort(df_seg, budget_map, new_hr)

        if before == 0:
            continue

        rows.append({
            "placement_group":  pg,
            "cohort_key":       ck,
            "n_rows":           n,
            "floor_price":      floor,
            "r_star":           r_star,
            "new_hr_applied":   new_hr,
            "ad_fee_before":    before,
            "ad_fee_after":     after,
            "revenue_lift_pct": (after - before) / before * 100,
            "avg_cpc_before":   before / n,
            "avg_cpc_after":    after / n,
        })
        print(
            f"  [{pg} / {ck}]  new_hr=${new_hr:.4f}  "
            f"before=${before:.2f}  after=${after:.2f}  "
            f"lift={rows[-1]['revenue_lift_pct']:+.4f}%"
        )

    return (
        pd.DataFrame(rows)
        .sort_values("revenue_lift_pct", ascending=False)
        .reset_index(drop=True)
    )


# ── ROAS computation ──────────────────────────────────────────────────────────
def compute_roas(
    summary: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach ROAS before/after columns to summary using per-auction attributed sales.

    eval_df must have an 'attributed_sales_usd' column (NaN for auctions with no sales)
    joined from FACT_ITEM_ORDER_ATTRIBUTION via auction_id.

    Before: sum(attributed_sales_usd) / sum(cpc_dollars)   [all auctions in cohort]
    After:  sum(attributed_sales for bid >= new_hr) / sum(new_cpc for bid >= new_hr)
            where new_cpc = min(bid, max(gsp, sr, new_hr))

    Raising the hard reserve causes some auctions to lose (bid < new_hr); their
    attributed sales are lost, and their CPC drops to zero.
    """
    roas_rows = []
    for _, row in summary.iterrows():
        pg     = row["placement_group"]
        ck     = row["cohort_key"]
        new_hr = float(row["new_hr_applied"])

        cohort_df = eval_df[
            (eval_df["placement_group"] == pg) & (eval_df["cohort_key"] == ck)
        ]
        if cohort_df.empty:
            continue

        bid   = cohort_df["auction_bid_dollars"].to_numpy(dtype=float)
        gsp   = cohort_df["raw_gsp_dollars"].to_numpy(dtype=float)
        sr    = cohort_df["soft_reserve_dollars"].to_numpy(dtype=float)
        cpc   = cohort_df["cpc_dollars"].to_numpy(dtype=float)
        sales = cohort_df["attributed_sales_usd"].fillna(0.0).to_numpy(dtype=float)

        # Before: all clicked winners, actual CPC and sales
        ad_fee_before = cpc.sum()
        if ad_fee_before == 0:
            continue
        roas_before = sales.sum() / ad_fee_before

        # After: only auctions where bid >= new_hr remain winners
        wins_after = bid >= new_hr
        new_cpc    = np.where(
            wins_after,
            np.minimum(bid, np.maximum(np.maximum(gsp, sr), new_hr)),
            0.0,
        )
        ad_fee_after  = new_cpc.sum()
        sales_after   = sales[wins_after].sum()
        roas_after    = sales_after / ad_fee_after if ad_fee_after > 0 else 0.0

        roas_rows.append({
            "placement_group": pg,
            "cohort_key":      ck,
            "roas_before":     round(roas_before, 4),
            "roas_after":      round(roas_after,  4),
            "roas_change":     round(roas_after - roas_before, 4),
        })

    return summary.merge(pd.DataFrame(roas_rows), on=["placement_group", "cohort_key"], how="left")


# ── Plot helpers ──────────────────────────────────────────────────────────────
TOP_N = 20


def _top_cohorts(summary: pd.DataFrame, pg: str, n: int = TOP_N) -> pd.DataFrame:
    return (
        summary[summary["placement_group"] == pg]
        .nlargest(n, "revenue_lift_pct")
        .reset_index(drop=True)
    )


def plot_revenue_lift(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    fig.suptitle(
        f"Revenue Lift by Cohort — Myerson Optimal Reserve (top {TOP_N} per group)\n"
        f"(SP clicked winners, budget-aware, eval={EVAL_START_DATE}–{EVAL_END_DATE})",
        fontsize=15,
    )
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        top = _top_cohorts(summary, pg)
        if top.empty:
            ax.set_title(f"{pg}\n(no data)")
            continue
        labels = top["cohort_key"].astype(str).tolist()[::-1]
        lifts  = top["revenue_lift_pct"].tolist()[::-1]
        hr_vals = top["new_hr_applied"].tolist()[::-1]
        y_pos  = np.arange(len(labels))
        bars = ax.barh(y_pos, lifts, color="steelblue", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Revenue Lift (%)")
        ax.set_title(f"{pg}\n(dim: {COHORT_DIM.get(pg, pg)})", fontsize=12)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        for bar, hr in zip(bars, hr_vals):
            ax.text(
                bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"HR=${hr:.2f}", va="center", fontsize=7, color="darkred",
            )
    plt.tight_layout()
    plt.show()


def plot_roas(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    fig.suptitle(
        f"ROAS Before vs After — Myerson Optimal Reserve (top {TOP_N} per group)\n"
        f"(SP clicked winners, budget-aware, eval={EVAL_START_DATE}–{EVAL_END_DATE})",
        fontsize=15,
    )
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        top = _top_cohorts(summary, pg).dropna(subset=["roas_before"])
        if top.empty:
            ax.set_title(f"{pg}\n(no data)")
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
        ax.set_title(pg, fontsize=12)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cpc(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    fig.suptitle(
        f"Avg CPC Before vs After — Myerson Optimal Reserve (top {TOP_N} per group)\n"
        f"(SP clicked winners, budget-aware, eval={EVAL_START_DATE}–{EVAL_END_DATE})",
        fontsize=15,
    )
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        top = _top_cohorts(summary, pg)
        if top.empty:
            ax.set_title(f"{pg}\n(no data)")
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
        ax.set_title(pg, fontsize=12)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_optimal_reserves(optimal_hr_map: dict) -> None:
    """Histogram of Myerson-optimal r* values per placement group."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Distribution of Myerson Optimal Reserve Prices per Placement Group", fontsize=13)
    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        floor = FLOOR_PRICES[pg]
        r_stars = [v for (g, _), v in optimal_hr_map.items() if g == pg]
        if not r_stars:
            ax.set_title(f"{pg}\n(no cohorts)")
            continue
        ax.hist(r_stars, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(floor, color="red", linestyle="--", lw=1.5, label=f"Floor=${floor:.2f}")
        ax.set_xlabel("Optimal Reserve r* ($)")
        ax.set_ylabel("Cohort count")
        ax.set_title(f"{pg}\n({len(r_stars)} cohorts)", fontsize=12)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
#%% Training: fetch auction candidates from clicked auctions over training window
print(f"Training window: {TRAIN_START_DATE} – {TRAIN_END_DATE}")
print(f"Distribution: {DIST_TYPE}  |  MIN_COHORT_BIDS={MIN_COHORT_BIDS}  |  MAX_RANK={MAX_RANK}")

train_df = fetch_train_data()
train_df.to_pickle(f"data/simulation_ctx_train_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}_smpl_{TRAIN_SAMPLE_PCT}_df.pkl")

# train_df = pd.read_pickle("data/simulation_ctx_train_df.pkl")

print(f"  Training rows: {len(train_df):,}")

#%% Fit distributions and solve Myerson's equation per cohort
print("\nFitting distributions and solving for Myerson optimal reserves...")
optimal_hr_map = train_optimal_reserves(train_df)

# Summarise r* by placement group
print("\nOptimal reserve summary by placement group:")
for pg in PLACEMENT_GROUP_ORDER:
    r_stars = [v for (g, _), v in optimal_hr_map.items() if g == pg]
    if r_stars:
        print(
            f"  {pg}: {len(r_stars)} cohorts  "
            f"r* range=[${min(r_stars):.3f}, ${max(r_stars):.3f}]  "
            f"median=${np.median(r_stars):.3f}"
        )

#%% Evaluation: fetch clicked winners from eval date range
print(f"\nFetching evaluation data ({EVAL_START_DATE} – {EVAL_END_DATE})...")

eval_df = fetch_eval_data()
eval_df.to_pickle(f"data/simulation_ctx_eval_{EVAL_START_DATE}_to_{EVAL_END_DATE}_smpl_{EVAL_SAMPLE_PCT}_df.pkl")

# eval_df = pd.read_pickle("data/simulation_ctx_eval_df.pkl")

print(f"  Eval clicked winners: {len(eval_df):,}")
print(f"  Eval total CPC ($):   {eval_df['cpc_dollars'].sum():,.2f}")

#%% Fetch budget and ROAS data
print(f"\nFetching campaign daily budgets for {EVAL_END_DATE}...")

budget_df = fetch_budget()
budget_df.to_pickle("data/simulation_ctx_budget_df.pkl")

# budget_df = pd.read_pickle("data/simulation_ctx_budget_df.pkl")

budget_map = budget_df.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
print(f"  Campaigns with budget: {len(budget_map):,}")

print(f"\nFetching per-auction attributed sales ({EVAL_START_DATE} – {EVAL_END_DATE})...")

sales_df = fetch_sales()
sales_df.to_pickle(f"data/simulation_ctx_sales_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

# sales_df = pd.read_pickle(f"data/simulation_ctx_sales_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

print(f"  Auctions with sales: {len(sales_df):,}")

#%% Run evaluation replay
print("\nRunning auction replay with Myerson-optimal reserves...")
# Add cohort columns and join per-auction sales for ROAS computation downstream
eval_df["placement_group"] = eval_df["placement"].map(PLACEMENT_TO_GROUP).fillna("Other")
eval_df["hour_bucket"] = eval_df["hour_bucket"].astype(str)
conditions = [eval_df["placement_group"] == pg for pg in PLACEMENT_GROUP_ORDER]
choices    = [eval_df[COHORT_DIM[pg]] for pg in PLACEMENT_GROUP_ORDER]
eval_df["cohort_key"] = np.select(conditions, choices, default="Other")
eval_df = eval_df.merge(sales_df, on="auction_id", how="left")

summary = evaluate_all_cohorts(eval_df, budget_map, optimal_hr_map)

#%% Compute ROAS before/after
summary = compute_roas(summary, eval_df)

#%% Print summary table
print("\nRevenue Lift by (Placement Group, Cohort) — top per group:")
header = f"{'Cohort Key':<40} {'r* ($)':>8} {'HR Applied':>11} {'Lift (%)':>10}"
for pg in PLACEMENT_GROUP_ORDER:
    top = _top_cohorts(summary, pg)
    print(f"\n  ── {pg} {'─' * 65}")
    print(f"  {header}")
    for _, row in top.iterrows():
        r_str = f"${row['r_star']:.4f}" if pd.notna(row.get("r_star")) else "N/A"
        print(
            f"  {str(row['cohort_key']):<40}"
            f" {r_str:>8} ${row['new_hr_applied']:.4f}  {row['revenue_lift_pct']:>+10.4f}%"
        )

total_lift_dollars = (summary["ad_fee_after"] - summary["ad_fee_before"]).sum()
total_cpc = summary["ad_fee_before"].sum()
overall_lift_pct = total_lift_dollars / total_cpc * 100 if total_cpc > 0 else 0.0
print(f"\n{'─' * 97}")
print(
    f"Overall revenue lift (Myerson optimal HR per cohort): {overall_lift_pct:.4f}%"
    f"  (${total_lift_dollars:,.2f} lift on ${total_cpc:,.2f} total CPC)"
)

#%% Plots
plot_optimal_reserves(optimal_hr_map)
plot_revenue_lift(summary)
plot_roas(summary)
plot_cpc(summary)

#%% Monetization rate before / after
# Monetization rate = sum(CPC paid) / sum(bid)
# Build a per-row lookup: (placement_group, cohort_key) -> new_hr_applied
hr_lookup = (
    summary.set_index(["placement_group", "cohort_key"])["new_hr_applied"].to_dict()
)

new_hr_vec = pd.Series(
    [hr_lookup.get((pg, ck), FLOOR_PRICES.get(pg, 0.0))
     for pg, ck in zip(eval_df["placement_group"], eval_df["cohort_key"])],
    index=eval_df.index,
)

bid = eval_df["auction_bid_dollars"]
gsp = eval_df["raw_gsp_dollars"]
sr  = eval_df["soft_reserve_dollars"]
cpc = eval_df["cpc_dollars"]

new_cpc_vec = np.where(
    bid < new_hr_vec,
    0.0,
    np.minimum(bid, np.maximum.reduce([gsp.values, sr.values, new_hr_vec.values])),
)

# ── Global monetization rate ──────────────────────────────────────────────────
total_bid = bid.sum()
mr_before_global = cpc.sum() / total_bid if total_bid > 0 else np.nan
mr_after_global  = new_cpc_vec.sum() / total_bid if total_bid > 0 else np.nan

print(f"\n{'─' * 55}")
print(f"{'Monetization Rate (MR = CPC / bid)':^55}")
print(f"{'─' * 55}")
print(f"  Global MR before: {mr_before_global:.4%}")
print(f"  Global MR after:  {mr_after_global:.4%}")
print(f"  Global MR delta:  {mr_after_global - mr_before_global:+.4%}")
print(f"{'─' * 55}")

# ── Per-cohort monetization rate ──────────────────────────────────────────────
eval_df["new_cpc"] = new_cpc_vec

cohort_mr = (
    eval_df.groupby(["placement_group", "cohort_key"])
    .apply(lambda g: pd.Series({
        "bid_sum":       g["auction_bid_dollars"].sum(),
        "cpc_sum":       g["cpc_dollars"].sum(),
        "new_cpc_sum":   g["new_cpc"].sum(),
    }))
    .reset_index()
)
cohort_mr["mr_before"] = cohort_mr["cpc_sum"]     / cohort_mr["bid_sum"]
cohort_mr["mr_after"]  = cohort_mr["new_cpc_sum"] / cohort_mr["bid_sum"]
cohort_mr["mr_delta"]  = cohort_mr["mr_after"] - cohort_mr["mr_before"]

# Merge with summary for new_hr_applied / r_star
cohort_mr = cohort_mr.merge(
    summary[["placement_group", "cohort_key", "new_hr_applied", "r_star", "n_rows"]],
    on=["placement_group", "cohort_key"],
    how="left",
)
cohort_mr = cohort_mr.sort_values(["placement_group", "mr_delta"], ascending=[True, False])

# Print per-placement-group table
hdr = f"{'Cohort Key':<40} {'HR ($)':>8} {'MR Before':>10} {'MR After':>10} {'Delta':>8}"
for pg in PLACEMENT_GROUP_ORDER:
    sub = cohort_mr[cohort_mr["placement_group"] == pg].head(10)
    if sub.empty:
        continue
    print(f"\n  ── {pg} {'─' * 63}")
    print(f"  {hdr}")
    for _, row in sub.iterrows():
        hr_str = f"${row['new_hr_applied']:>6.4f}" if pd.notna(row.get("new_hr_applied")) else "   N/A  "
        print(
            f"  {str(row['cohort_key']):<40}"
            f" {hr_str}"
            f" {row['mr_before']:>10.4%}"
            f" {row['mr_after']:>10.4%}"
            f" {row['mr_delta']:>+8.4%}"
        )

# ── Plot: MR before / after per placement group ───────────────────────────────
def plot_monetization_rate(cohort_mr, top_n=15):
    fig, axes = plt.subplots(
        1, len(PLACEMENT_GROUP_ORDER),
        figsize=(5 * len(PLACEMENT_GROUP_ORDER), 6),
    )
    if len(PLACEMENT_GROUP_ORDER) == 1:
        axes = [axes]

    for ax, pg in zip(axes, PLACEMENT_GROUP_ORDER):
        sub = (
            cohort_mr[cohort_mr["placement_group"] == pg]
            .nlargest(top_n, "mr_delta")
            .sort_values("mr_delta")
        )
        if sub.empty:
            ax.set_visible(False)
            continue

        labels = [str(ck)[:30] for ck in sub["cohort_key"]]
        y = np.arange(len(labels))
        bar_h = 0.35

        ax.barh(y + bar_h / 2, sub["mr_before"], bar_h, label="MR Before", color="steelblue", alpha=0.8)
        ax.barh(y - bar_h / 2, sub["mr_after"],  bar_h, label="MR After",  color="darkorange", alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.set_xlabel("Monetization Rate (CPC / bid)")
        ax.set_title(f"{pg} — Top {top_n} MR-increasing cohorts")
        ax.legend(fontsize=8)

    plt.suptitle("Monetization Rate Before vs After Myerson Optimal Hard Reserve", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

plot_monetization_rate(cohort_mr)
