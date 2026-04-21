import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from scipy.optimize import minimize, brentq
from scipy.stats import gamma as gamma_dist, lognorm

plt.close("all")

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_END_DATE      = "2026-03-25"   # training window end (inclusive)
TRAIN_N_DAYS        = 7              # days to look back from TRAIN_END_DATE
EVAL_DATE           = "2026-04-01"   # future date for auction replay evaluation
SAMPLE_PCT          = 100            # campaign-level sampling (MOD HASH < SAMPLE_PCT)
MAX_RANK            = 5              # use auction_rank < MAX_RANK for training bids
MIN_COHORT_BIDS     = 1000           # min bid rows per cohort to fit a distribution
DIST_TYPE           = "gamma"        # "gamma" or "lognormal"
SELLER_VALUE        = 0.0            # Myerson seller valuation (v_0), usually 0
ROAS_SNAPSHOT_START = "2026-03-19"
ROAS_SNAPSHOT_END   = "2026-03-25"

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
  AND MOD(ABS(HASH(acd.campaign_id)), 100) < {sample_pct}
"""

# ── Evaluation SQL ────────────────────────────────────────────────────────────
# Clicked winners (rank=0) on EVAL_DATE with all pricing and cohort fields.
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
        COALESCE(acd.collection_id, 'Unknown')                                     AS collection_id
    FROM edw.ads.ads_auction_candidates_event_delta acd
    WHERE acd.event_date = '{eval_date}'
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
    WHERE event_date = '{eval_date}'
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


# ── Data fetchers ─────────────────────────────────────────────────────────────
def _train_date_range(end_date: str = TRAIN_END_DATE, n_days: int = TRAIN_N_DAYS):
    end = date.fromisoformat(end_date)
    start = end - timedelta(days=n_days - 1)
    return start.isoformat(), end.isoformat()


def fetch_train_data(
    train_end_date: str = TRAIN_END_DATE,
    n_days: int = TRAIN_N_DAYS,
    sample_pct: int = SAMPLE_PCT,
    max_rank: int = MAX_RANK,
) -> pd.DataFrame:
    train_start, train_end = _train_date_range(train_end_date, n_days)
    query = TRAINING_QUERY.format(
        train_start_date=train_start,
        train_end_date=train_end,
        max_rank=max_rank,
        sample_pct=sample_pct,
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


def fetch_eval_data(eval_date: str = EVAL_DATE, sample_pct: int = SAMPLE_PCT) -> pd.DataFrame:
    query = EVAL_QUERY.format(eval_date=eval_date, sample_pct=sample_pct)
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


def fetch_budget(budget_date: str = EVAL_DATE) -> pd.DataFrame:
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


def fetch_roas(
    start_date: str = ROAS_SNAPSHOT_START,
    end_date: str = ROAS_SNAPSHOT_END,
) -> pd.DataFrame:
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

    For each (placement_group, cohort_key) that appears in eval_df:
      - Look up r_star from optimal_hr_map.
      - Only apply r_star if r_star > FLOOR_PRICES[pg]; otherwise no change.

    Returns a summary DataFrame per cohort with before/after metrics.
    """
    df = _add_cohort_columns(eval_df)
    df = df[df["placement_group"].isin(PLACEMENT_GROUP_ORDER)].copy()
    # hour_bucket for DoubleDash comes from event_timestamp in eval data
    df["hour_bucket"] = df["event_timestamp"].dt.hour.astype(str)
    # Recompute cohort_key with the corrected hour_bucket
    conditions = [df["placement_group"] == pg for pg in PLACEMENT_GROUP_ORDER]
    choices    = [df[COHORT_DIM[pg]] for pg in PLACEMENT_GROUP_ORDER]
    df["cohort_key"] = np.select(conditions, choices, default="Other")

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
    roas_df: pd.DataFrame,
    roas_start: str = ROAS_SNAPSHOT_START,
    roas_end: str = ROAS_SNAPSHOT_END,
) -> pd.DataFrame:
    """
    Attach ROAS before/after columns to summary (mirrors segment_placement_ctx.py).
    Attribution is proportional by each campaign's CPC share in the cohort.
    Both sales and ad_fee are normalised to a per-day basis.
    """
    n_days = (date.fromisoformat(roas_end) - date.fromisoformat(roas_start)).days + 1
    roas_lookup = roas_df.set_index("campaign_id")
    campaign_total_cpc = eval_df.groupby("campaign_id")["cpc_dollars"].sum()

    roas_rows = []
    for _, row in summary.iterrows():
        pg = row["placement_group"]
        ck = row["cohort_key"]

        cohort_df = eval_df[(eval_df["placement_group"] == pg) & (eval_df["cohort_key"] == ck)]
        cohort_cpc = cohort_df.groupby("campaign_id")["cpc_dollars"].sum()
        fractions = (cohort_cpc / campaign_total_cpc.reindex(cohort_cpc.index)).fillna(1.0)

        cr = roas_lookup.reindex(fractions.index).dropna(subset=["total_ad_fee_usd"])
        cr = cr.join(fractions.rename("fraction"), how="left").fillna({"fraction": 1.0})

        if cr.empty or (cr["total_ad_fee_usd"] * cr["fraction"]).sum() == 0:
            continue

        sales    = (cr["total_attributed_sales_usd"] * cr["fraction"]).sum() / n_days
        ad_fee   = (cr["total_ad_fee_usd"]           * cr["fraction"]).sum() / n_days
        lift     = row["ad_fee_after"] - row["ad_fee_before"]

        roas_before = sales / ad_fee
        roas_after  = sales / (ad_fee + lift) if (ad_fee + lift) > 0 else 0.0

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
        f"(SP clicked winners, budget-aware, eval={EVAL_DATE})",
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
        f"(SP clicked winners, budget-aware, eval={EVAL_DATE})",
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
        f"(SP clicked winners, budget-aware, eval={EVAL_DATE})",
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
#%% Training: fetch auction candidates from clicked auctions over N-day window
train_start, train_end = _train_date_range()
print(f"Training window: {train_start} – {train_end}  ({TRAIN_N_DAYS} days)")
print(f"Distribution: {DIST_TYPE}  |  MIN_COHORT_BIDS={MIN_COHORT_BIDS}  |  MAX_RANK={MAX_RANK}")

# train_df = fetch_train_data()
# train_df.to_pickle("data/simulation_ctx_train_df.pkl")
train_df = pd.read_pickle("data/simulation_ctx_train_df.pkl")
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

#%% Evaluation: fetch clicked winners from future eval date
print(f"\nFetching evaluation data for {EVAL_DATE}...")
# eval_df = fetch_eval_data()
# eval_df.to_pickle("data/simulation_ctx_eval_df.pkl")
eval_df = pd.read_pickle("data/simulation_ctx_eval_df.pkl")
print(f"  Eval clicked winners: {len(eval_df):,}")
print(f"  Eval total CPC ($):   {eval_df['cpc_dollars'].sum():,.2f}")

#%% Fetch budget and ROAS data
print(f"\nFetching campaign daily budgets for {EVAL_DATE}...")
# budget_df = fetch_budget()
# budget_df.to_pickle("data/simulation_ctx_budget_df.pkl")
budget_df = pd.read_pickle("data/simulation_ctx_budget_df.pkl")
budget_map = budget_df.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
print(f"  Campaigns with budget: {len(budget_map):,}")

print(f"\nFetching ROAS data ({ROAS_SNAPSHOT_START} – {ROAS_SNAPSHOT_END})...")
# roas_df = fetch_roas()
# roas_df.to_pickle("data/simulation_ctx_roas_df.pkl")
roas_df = pd.read_pickle("data/simulation_ctx_roas_df.pkl")
print(f"  Campaigns with ROAS:   {len(roas_df):,}")

#%% Run evaluation replay
print("\nRunning auction replay with Myerson-optimal reserves...")
# Add cohort columns to eval_df for ROAS computation downstream
eval_df["placement_group"] = eval_df["placement"].map(PLACEMENT_TO_GROUP).fillna("Other")
eval_df["hour_bucket"] = eval_df["event_timestamp"].dt.hour.astype(str)
conditions = [eval_df["placement_group"] == pg for pg in PLACEMENT_GROUP_ORDER]
choices    = [eval_df[COHORT_DIM[pg]] for pg in PLACEMENT_GROUP_ORDER]
eval_df["cohort_key"] = np.select(conditions, choices, default="Other")

summary = evaluate_all_cohorts(eval_df, budget_map, optimal_hr_map)

#%% Compute ROAS before/after
summary = compute_roas(summary, eval_df, roas_df)

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
