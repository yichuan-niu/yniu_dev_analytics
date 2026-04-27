"""Data fetching, distribution fitting, and auction simulation functions."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import gammainc as _gammainc, gammaln as _gammaln, digamma as _digamma, ndtr as _ndtr
from scipy.stats import gamma as gamma_dist, lognorm

from simulation_customized_ctx_config import (
    PLACEMENT_TO_GROUP,
    PLACEMENT_GROUP_ORDER,
    COHORT_DIM,
    FLOOR_PRICES,
    get_connection,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    EVAL_START_DATE,
    EVAL_END_DATE,
    TRAIN_SAMPLE_PCT,
    EVAL_SAMPLE_PCT,
    MAX_RANK,
    MIN_COHORT_BIDS,
    TOP_N_COHORTS,
    DIST_TYPE,
    LOGNORM_SIGMA_MAX,
    SELLER_VALUE,
    MAX_RESERVE_INC,
    L1_CATEGORY_NAMES,
)


def _display_cohort_key(pg: str, ck: str) -> str:
    """Return a human-readable cohort key (resolve category IDs to names)."""
    if pg == "Category":
        return L1_CATEGORY_NAMES.get(str(ck), str(ck))
    return str(ck)


# ── Training SQL ──────────────────────────────────────────────────────────────
# Pulls all auction candidates (rank < MAX_RANK) over the training window.
# Uses all auctions (not just clicked) for unbiased bid distribution fitting.
# Floor filtering is done in Python per group.
# event_hour from the auction table is used as hour_bucket for DoubleDash.
TRAINING_QUERY = """
SELECT
    acd.placement,
    acd.auction_bid / 100.0                                             AS auction_bid_dollars,
    GET(PARSE_JSON(acd.pricing_metadata), 'hardReserve')::INT / 100.0  AS hard_reserve_dollars,
    COALESCE(acd.normalized_query, 'Unknown')                          AS normalized_query,
    COALESCE(acd.l1_category_id::VARCHAR, 'Unknown')                   AS l1_category_id,
    COALESCE(acd.collection_id, 'Unknown')                             AS collection_id,
    acd.event_hour                                                      AS hour_bucket
FROM edw.ads.ads_auction_candidates_event_delta acd
WHERE acd.event_date BETWEEN '{train_start_date}' AND '{train_end_date}'
  AND acd.CURRENCY_ISO_TYPE IN ('USD')
  AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
  AND acd.auction_rank < {max_rank}
  AND acd.pricing_metadata IS NOT NULL
  AND MOD(ABS(HASH(acd.auction_id)), 1000000) < {train_sample_pct} * 10000
"""

# ── Evaluation SQL ────────────────────────────────────────────────────────────
# All candidates (rank < max_rank) for clicked SP USD auctions over the eval
# date range.  The rank-0 winner must have pricing_metadata for baseline CPC.
# All other candidates are included (even without pricing_metadata) so that
# runner-up promotion and GSP computation work correctly.
EVAL_QUERY = """
WITH clicked AS (
    SELECT
        ad_auction_id
    FROM proddb.public.fact_item_card_click_dedup
    WHERE event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
      AND is_sponsored = 1
      AND is_cpc = 1
      AND ad_auction_id IS NOT NULL
      AND campaign_id IS NOT NULL
    GROUP BY ad_auction_id
),
base AS (
    SELECT
        acd.auction_id,
        acd.campaign_id,
        acd.placement,
        acd.event_date,
        acd.auction_rank,
        acd.auction_bid / 100.0                                                          AS auction_bid_dollars,
        acd.ad_score / 100.0                                                             AS ad_score_dollars,
        COALESCE(GET(PARSE_JSON(acd.pricing_metadata), 'hardReserve')::INT, 0) / 100.0   AS hard_reserve_dollars,
        COALESCE(GET(PARSE_JSON(acd.pricing_metadata), 'softReserveBeta')::FLOAT, 0)     AS soft_reserve_beta,
        COALESCE(acd.normalized_query, 'Unknown')                                        AS normalized_query,
        COALESCE(acd.l1_category_id::VARCHAR, 'Unknown')                                 AS l1_category_id,
        COALESCE(acd.collection_id, 'Unknown')                                           AS collection_id,
        acd.event_hour                                                                    AS hour_bucket,
        acd.occurred_at                                                                   AS auction_timestamp,
        MAX(CASE
            WHEN acd.auction_rank = 0
             AND acd.pricing_metadata IS NOT NULL
             AND MOD(ABS(HASH(acd.campaign_id)), 100) < {eval_sample_pct}
            THEN 1 ELSE 0
        END) OVER (PARTITION BY acd.auction_id)                                           AS _qualifies
    FROM edw.ads.ads_auction_candidates_event_delta acd
    INNER JOIN clicked ON acd.auction_id = clicked.ad_auction_id
    WHERE acd.event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
      AND acd.CURRENCY_ISO_TYPE IN ('USD')
      AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
      AND acd.auction_rank < {max_rank}
)
SELECT
    auction_id, campaign_id, placement, event_date, auction_rank,
    auction_bid_dollars, ad_score_dollars, hard_reserve_dollars,
    soft_reserve_beta, normalized_query, l1_category_id, collection_id,
    hour_bucket, auction_timestamp
FROM base
WHERE _qualifies = 1
"""

BUDGET_QUERY = """
SELECT
    date_est,
    campaign_id,
    COALESCE(MAX(campaign_budget), SUM(daily_budget)) / 100 AS campaign_daily_budget_dollars
FROM PRODDB.PUBLIC.FACT_ADS_DAILY_BUDGET
WHERE date_est BETWEEN '{start_date}' AND '{end_date}'
  AND CURRENCY = 'USD'
GROUP BY date_est, campaign_id
"""

SALES_QUERY = """
SELECT
    AD_AUCTION_ID                                       AS auction_id,
    SUM(PRICE_UNIT_AMOUNT * QUANTITY_RECEIVED) / 100.0 AS attributed_sales_usd
FROM proddb.public.FACT_ITEM_ORDER_ATTRIBUTION
WHERE SNAPSHOT_DATE BETWEEN DATEADD(day, -1, '{eval_start_date}'::DATE) AND DATEADD(day, 1, '{eval_end_date}'::DATE)
  AND EVENT_TIMESTAMP >= '{eval_start_date}' AND EVENT_TIMESTAMP < DATEADD(day, 1, '{eval_end_date}'::DATE)
GROUP BY AD_AUCTION_ID
"""


# ── Data fetchers ─────────────────────────────────────────────────────────────
def fetch_train_data(
    train_start_date: str = TRAIN_START_DATE,
    train_end_date: str = TRAIN_END_DATE,
    train_sample_pct: float = TRAIN_SAMPLE_PCT,
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
    return df


def fetch_eval_data(
    eval_start_date: str = EVAL_START_DATE,
    eval_end_date: str = EVAL_END_DATE,
    eval_sample_pct: int = EVAL_SAMPLE_PCT,
    max_rank: int = MAX_RANK,
) -> pd.DataFrame:
    query = EVAL_QUERY.format(
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
        eval_sample_pct=eval_sample_pct,
        max_rank=max_rank,
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    for col in ["auction_bid_dollars", "ad_score_dollars",
                "hard_reserve_dollars", "soft_reserve_beta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ad_score_dollars"] = df["ad_score_dollars"].fillna(0.0)
    df["soft_reserve_beta"] = df["soft_reserve_beta"].fillna(0.0)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date.astype(str)
    df["auction_timestamp"] = pd.to_datetime(df["auction_timestamp"], errors="coerce")
    df["hour_bucket"] = pd.to_numeric(df["hour_bucket"], errors="coerce").astype("Int64")
    df["auction_rank"] = pd.to_numeric(df["auction_rank"], errors="coerce").astype("Int64")
    df["auction_id"] = df["auction_id"].astype(str)
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df


def fetch_budget(
    start_date: str = EVAL_START_DATE,
    end_date: str = EVAL_END_DATE,
) -> pd.DataFrame:
    """Fetch per-day campaign budgets for the eval date range."""
    query = BUDGET_QUERY.format(start_date=start_date, end_date=end_date)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df["date_est"] = pd.to_datetime(df["date_est"], errors="coerce").dt.date.astype(str)
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
def _gamma_nll_and_grad(params, sum_log_bids, sum_bids, n, floor, log_floor):
    """NLL and gradient for Gamma truncated from below at floor."""
    alpha, theta = params
    if alpha <= 0 or theta <= 0:
        return 1e18, np.array([0.0, 0.0])
    u = floor / theta
    surv = 1.0 - _gammainc(alpha, u)
    if surv <= 0:
        return 1e18, np.array([0.0, 0.0])
    log_theta = np.log(theta)
    log_surv = np.log(surv)

    # vectorized NLL using precomputed sufficient statistics
    nll = (-(alpha - 1.0) * sum_log_bids
           + sum_bids / theta
           + n * alpha * log_theta
           + n * _gammaln(alpha)
           + n * log_surv)

    # pdf at floor (log-space for stability)
    log_pdf_floor = ((alpha - 1.0) * log_floor - floor / theta
                     - alpha * log_theta - _gammaln(alpha))
    pdf_floor = np.exp(log_pdf_floor)

    # d(log surv)/d(theta): analytical
    d_log_surv_dtheta = (floor / theta) * pdf_floor / surv

    # d(log surv)/d(alpha): central finite difference on gammainc
    eps = 1e-7 * max(abs(alpha), 1.0)
    d_gammainc_da = (_gammainc(alpha + eps, u) - _gammainc(alpha - eps, u)) / (2.0 * eps)
    d_log_surv_dalpha = -d_gammainc_da / surv

    grad_alpha = -sum_log_bids + n * log_theta + n * _digamma(alpha) + n * d_log_surv_dalpha
    grad_theta = -sum_bids / theta**2 + n * alpha / theta + n * d_log_surv_dtheta

    return nll, np.array([grad_alpha, grad_theta])


def fit_gamma_truncated(bids: np.ndarray, floor: float):
    """Fit a Gamma distribution to bids truncated from below at floor via L-BFGS-B."""
    m = bids.mean()
    v = bids.var()
    v = v if v > 0 else m
    theta0 = v / m
    alpha0 = m / theta0
    # precompute sufficient statistics
    n = len(bids)
    log_bids = np.log(bids)
    sum_log_bids = log_bids.sum()
    sum_bids = bids.sum()
    log_floor = np.log(floor)
    result = minimize(
        _gamma_nll_and_grad,
        x0=[alpha0, theta0],
        args=(sum_log_bids, sum_bids, n, floor, log_floor),
        method="L-BFGS-B",
        jac=True,
        bounds=[(1e-3, None), (1e-6, None)],
        options={"ftol": 1e-10, "gtol": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"Gamma MLE failed to converge: {result.message}")
    alpha, theta = result.x
    return gamma_dist(a=alpha, scale=theta)


_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _lognorm_nll_and_grad(params, sum_log_bids, sum_log_bids_sq, n, log_floor):
    """NLL and analytical gradient for Lognormal truncated from below at floor."""
    mu, sigma = params
    if sigma <= 0:
        return 1e18, np.array([0.0, 0.0])
    z_f = (log_floor - mu) / sigma
    surv = 1.0 - _ndtr(z_f)       # survival = 1 - Phi(z_f)
    if surv <= 0:
        return 1e18, np.array([0.0, 0.0])

    SS = sum_log_bids_sq - 2.0 * mu * sum_log_bids + n * mu**2
    log_surv = np.log(surv)

    # vectorized NLL using precomputed sufficient statistics
    nll = (sum_log_bids + n * np.log(sigma) + n * 0.5 * np.log(2.0 * np.pi)
           + SS / (2.0 * sigma**2) + n * log_surv)

    # standard normal pdf at z_f
    phi_zf = _INV_SQRT_2PI * np.exp(-0.5 * z_f**2)
    trunc = n * phi_zf / (sigma * surv)

    grad_mu = (n * mu - sum_log_bids) / sigma**2 + trunc
    grad_sigma = n / sigma - SS / sigma**3 + trunc * z_f

    return nll, np.array([grad_mu, grad_sigma])


def fit_lognormal_truncated(
    bids: np.ndarray, floor: float, sigma_max: float = LOGNORM_SIGMA_MAX,
):
    """Fit a Lognormal distribution to bids truncated from below at floor via L-BFGS-B.

    sigma is bounded by sigma_max to ensure the virtual valuation stays monotone
    (required for a unique Myerson optimal reserve).
    """
    log_bids = np.log(bids)
    # precompute sufficient statistics
    n = len(bids)
    sum_log_bids = log_bids.sum()
    sum_log_bids_sq = (log_bids**2).sum()
    log_floor = np.log(floor)
    lb_std = log_bids.std()
    result = minimize(
        _lognorm_nll_and_grad,
        x0=[log_bids.mean(), min(lb_std if lb_std > 0 else 0.5, sigma_max)],
        args=(sum_log_bids, sum_log_bids_sq, n, log_floor),
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None), (1e-6, sigma_max)],
        options={"ftol": 1e-10, "gtol": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"Lognormal MLE failed to converge: {result.message}")
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
    hi: float = 10.0,
):
    """
    Find r* where ψ(r*) = seller_value using Brent's bisection method.

    Returns the raw r* (may be below floor or very large), or None if no root exists.
    """
    vv = lambda v: virtual_valuation(v, dist) - seller_value
    try:
        if vv(floor + 1e-6) >= 0:
            return None
        return brentq(vv, floor + 1e-6, hi, xtol=1e-2)
    except ValueError:
        return None


def clip_reserve(
    r_star,
    floor: float,
    max_increment: float = MAX_RESERVE_INC,
    label: str = "",
):
    """
    Validate and cap a raw Myerson reserve.

    Returns None (with a log message) if r* is missing or at/below floor,
    otherwise caps at floor + max_increment.
    """
    if r_star is None:
        return None
    if r_star <= floor:
        print(f"  {label}r*=${r_star:.4f} <= floor=${floor:.2f}, skipping")
        return None
    capped = min(r_star, floor + max_increment)
    if capped < r_star:
        print(f"  {label}r*=${r_star:.4f} capped to ${capped:.4f} (floor+{max_increment})")
    return capped


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
    top_n_cohorts: "int | None" = TOP_N_COHORTS,
) -> dict:
    """
    For each (placement_group, cohort_key) cohort with enough bid data:
      1. Filter bids > floor (removes floor-clamped auto-bid values).
      2. Fit Gamma (or lognormal) via truncated MLE.
      3. Solve Myerson's virtual valuation = 0 with Brent's method.
      4. Keep r* only if r* > floor.

    If top_n_cohorts is set, only the top-N cohorts by bid count per
    placement group are considered (after the min_cohort_bids filter).

    Returns
    -------
    optimal_hr : dict[(placement_group, cohort_key)] = r_star
    fitted_dists : dict[(placement_group, cohort_key)] = frozen scipy distribution
        The truncated-MLE fitted distribution used to derive each r*.
    """
    df = _add_cohort_columns(train_df)
    df = df[df["placement_group"].isin(PLACEMENT_GROUP_ORDER)].copy()

    optimal_hr: dict = {}
    fitted_dists: dict = {}
    skipped_small = skipped_solve = 0

    grouped = df.dropna(subset=["auction_bid_dollars", "hard_reserve_dollars"]).groupby(
        ["placement_group", "cohort_key"]
    )

    # Apply top-N filter per placement group (by bid count, descending)
    if top_n_cohorts is not None:
        bid_counts = {
            (pg, ck): (rows["auction_bid_dollars"] > rows["hard_reserve_dollars"]).sum()
            for (pg, ck), rows in grouped
        }
        allowed = set()
        for pg in PLACEMENT_GROUP_ORDER:
            pg_cohorts = sorted(
                [(ck, cnt) for (g, ck), cnt in bid_counts.items()
                 if g == pg and cnt >= min_cohort_bids],
                key=lambda x: x[1], reverse=True,
            )
            allowed.update((pg, ck) for ck, _ in pg_cohorts[:top_n_cohorts])
    else:
        allowed = None

    n_cohorts = len(grouped)
    print(f"  Total cohorts: {n_cohorts}")
    for i, ((pg, ck), cohort_rows) in enumerate(grouped, 1):
        floor = FLOOR_PRICES[pg]
        # Use per-row hard reserve for filtering (handles non-standard HR
        # from experiments or DV overrides within the training window)
        bids = cohort_rows.loc[
            cohort_rows["auction_bid_dollars"] > cohort_rows["hard_reserve_dollars"],
            "auction_bid_dollars",
        ].to_numpy(dtype=float)

        if allowed is not None and (pg, ck) not in allowed:
            continue

        if len(bids) < min_cohort_bids:
            skipped_small += 1
            # print(f"  [{pg} / {ck}]  skipped (n={len(bids):,} < {min_cohort_bids:,})")
            continue

        try:
            dist = fit_distribution(bids, floor, dist_type)
            r_raw = myerson_optimal_reserve(dist, floor)
            r_star = clip_reserve(r_raw, floor, label=f"[{pg} / {_display_cohort_key(pg, ck)}] ")
        except Exception:
            skipped_solve += 1
            continue

        if r_star is None:
            skipped_solve += 1
            continue

        optimal_hr[(pg, ck)] = r_star
        fitted_dists[(pg, ck)] = dist
        pct = i / n_cohorts * 100
        ck_label = _display_cohort_key(pg, ck)
        print(f"  [{pg} / {ck_label} ({ck})]  floor=${floor:.2f}  r*=${r_star:.4f}  n={len(bids):,}  ({pct:.0f}%)")

    print(f"\n  Cohorts solved: {len(optimal_hr)}")
    print(f"  Skipped (too few bids): {skipped_small}")
    print(f"  Skipped (no root / fit error): {skipped_solve}")
    return optimal_hr, fitted_dists


# ── Evaluation: auction resolution and budget-aware replay ────────────────────
def _compute_winner_cpc(
    candidates: pd.DataFrame,
    hr_col: str,
) -> pd.DataFrame:
    """
    Filter auction candidates by bid >= hard reserve, recompute GSP and soft
    reserve from raw ad_score, and return per-auction winner info.

    For each auction (after filtering):
      - raw_gsp   = next_ad_score / ad_score * bid
      - soft_reserve = soft_reserve_beta * next_bid
      - cpc       = min(bid, max(raw_gsp, soft_reserve, hr))
      where "next" refers to the next-ranked eligible candidate.

    Parameters
    ----------
    candidates : DataFrame with ALL candidates (multiple ranks per auction).
    hr_col     : column name holding the hard reserve to filter and price by.

    Returns
    -------
    One-row-per-auction DataFrame with columns:
        auction_id, campaign_id, auction_bid_dollars, cpc, auction_rank
    """
    df = candidates.sort_values(["auction_id", "auction_rank"]).copy()

    # Filter to eligible candidates
    eligible = df[df["auction_bid_dollars"] >= df[hr_col]].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=["auction_id", "campaign_id", "auction_bid_dollars",
                     "cpc", "auction_rank"]
        )

    # Next eligible candidate's values within each auction
    eligible["_next_bid"] = (
        eligible.groupby("auction_id")["auction_bid_dollars"]
        .shift(-1).fillna(0.0)
    )
    eligible["_next_ad_score"] = (
        eligible.groupby("auction_id")["ad_score_dollars"]
        .shift(-1).fillna(0.0)
    )

    bid = eligible["auction_bid_dollars"].to_numpy(dtype=float)
    ad_score = eligible["ad_score_dollars"].to_numpy(dtype=float)
    next_ad_score = eligible["_next_ad_score"].to_numpy(dtype=float)
    next_bid = eligible["_next_bid"].to_numpy(dtype=float)
    beta = eligible["soft_reserve_beta"].to_numpy(dtype=float)
    hr = eligible[hr_col].to_numpy(dtype=float)

    raw_gsp = np.where(ad_score > 0, next_ad_score / ad_score * bid, 0.0)
    soft_reserve = beta * next_bid
    comp = np.maximum(raw_gsp, soft_reserve)
    eligible["cpc"] = np.minimum(bid, np.maximum(comp, hr))

    # Winner = lowest auction_rank (highest priority) per auction
    winners = eligible.loc[
        eligible.groupby("auction_id")["auction_rank"].idxmin()
    ]
    return winners[["auction_id", "campaign_id", "auction_bid_dollars",
                    "cpc", "auction_rank"]].copy()


def resolve_auction_outcomes(
    eval_all: pd.DataFrame,
    optimal_hr_map: dict,
) -> pd.DataFrame:
    """
    For each clicked auction, determine baseline and new-scenario CPC
    by recomputing GSP and soft reserve from raw ad_score after filtering
    candidates by the respective hard reserve.

    Both scenarios use the same formula:
        final_cpc = min(bid, max(raw_gsp, soft_reserve, hard_reserve))
    where raw_gsp and soft_reserve are derived from the next eligible
    candidate's ad_score and bid after filtering.

    Parameters
    ----------
    eval_all : DataFrame with ALL candidates (multiple ranks per auction).
               Must already have placement_group and cohort_key columns.
    optimal_hr_map : dict[(placement_group, cohort_key)] -> r_star

    Returns
    -------
    One-row-per-auction DataFrame (the rank-0 row) with added columns:
        cpc_baseline, cpc_new, new_campaign_id, new_bid
    """
    df = eval_all.copy()

    # Cohort-aware new HR for every candidate row
    df["new_hr"] = [
        optimal_hr_map.get((pg, ck), hr)
        for pg, ck, hr in zip(
            df["placement_group"], df["cohort_key"], df["hard_reserve_dollars"]
        )
    ]

    # ── Baseline: filter by original HR, recompute CPC from ad_score ─────
    bl_winners = _compute_winner_cpc(df, "hard_reserve_dollars")

    # ── New scenario: filter by Myerson optimal HR, recompute CPC ────────
    nw_winners = _compute_winner_cpc(df, "new_hr")

    # ── Merge results onto rank-0 rows for context ───────────────────────
    rank0 = df[df["auction_rank"] == 0].copy()

    # Baseline CPC (winner should be rank-0 under original HR)
    rank0 = rank0.merge(
        bl_winners[["auction_id", "cpc"]].rename(columns={"cpc": "cpc_baseline"}),
        on="auction_id",
        how="left",
    )
    rank0["cpc_baseline"] = rank0["cpc_baseline"].fillna(0.0)

    # New scenario CPC, winner campaign, and winner bid
    rank0 = rank0.merge(
        nw_winners[["auction_id", "cpc", "campaign_id", "auction_bid_dollars"]]
        .rename(columns={
            "cpc": "cpc_new",
            "campaign_id": "new_campaign_id",
            "auction_bid_dollars": "new_bid",
        }),
        on="auction_id",
        how="left",
    )
    rank0["cpc_new"] = rank0["cpc_new"].fillna(0.0)
    rank0["new_bid"] = rank0["new_bid"].fillna(0.0)
    rank0["new_campaign_id"] = rank0["new_campaign_id"].fillna(
        rank0["campaign_id"]
    )

    n_promo = (rank0["new_campaign_id"] != rank0["campaign_id"]).sum()
    n_filtered = (rank0["cpc_new"] == 0.0).sum() - (rank0["cpc_baseline"] == 0.0).sum()
    print(f"  Runner-up promotions: {n_promo:,}")
    print(f"  Net auctions lost (no eligible candidate): {max(n_filtered, 0):,}")

    return rank0.drop(columns=["new_hr"])


def _apply_budget_caps(
    df: pd.DataFrame,
    budget_maps: dict[str, dict],
) -> None:
    """
    Apply global per-(date, campaign) budget caps in chronological order
    to pre-computed cpc_baseline and cpc_new columns.

    Baseline budgets are tracked by campaign_id (the original winner).
    New-scenario budgets are tracked by new_campaign_id (may differ when
    a runner-up from a different campaign is promoted).

    Adds columns to *df* in place: capped_baseline, capped_new.
    """
    df["capped_baseline"] = 0.0
    df["capped_new"] = 0.0

    for dt in sorted(budget_maps.keys()):
        day = df.loc[df["event_date"] == dt].sort_values(
            "auction_timestamp", na_position="last"
        )
        if day.empty:
            continue
        day_budget = budget_maps[dt]

        # ── Baseline: budget tracked by campaign_id ───────────────────────
        budgets_bl = day["campaign_id"].map(day_budget).fillna(float("inf"))
        cum_bl = day.groupby("campaign_id")["cpc_baseline"].cumsum()
        remaining_bl = (budgets_bl - (cum_bl - day["cpc_baseline"])).clip(lower=0)
        df.loc[day.index, "capped_baseline"] = (
            day["cpc_baseline"].clip(upper=remaining_bl).values
        )

        # ── New: budget tracked by new_campaign_id ────────────────────────
        budgets_nw = day["new_campaign_id"].map(day_budget).fillna(float("inf"))
        cum_nw = day.groupby("new_campaign_id")["cpc_new"].cumsum()
        remaining_nw = (budgets_nw - (cum_nw - day["cpc_new"])).clip(lower=0)
        df.loc[day.index, "capped_new"] = (
            day["cpc_new"].clip(upper=remaining_nw).values
        )


def evaluate_all_cohorts(
    eval_df: pd.DataFrame,
    optimal_hr_map: dict,
) -> pd.DataFrame:
    """
    Aggregate budget-capped replay results per cohort.

    Expects eval_df to already contain capped_baseline / capped_new columns
    written by _apply_budget_caps (global budget applied across all cohorts).
    Only reports cohorts where an optimal HR was found.
    """
    df = eval_df[eval_df["placement_group"].isin(PLACEMENT_GROUP_ORDER)]

    # Filter to cohorts where HR changed
    changed = set(optimal_hr_map.keys())
    mask = pd.Series(
        [(pg, ck) in changed
         for pg, ck in zip(df["placement_group"], df["cohort_key"])],
        index=df.index,
    )

    agg = (
        df[mask]
        .groupby(["placement_group", "cohort_key"])
        .agg(
            n_rows=("cpc_baseline", "size"),
            ad_fee_before=("capped_baseline", "sum"),
            ad_fee_after=("capped_new", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["ad_fee_before"] > 0].copy()

    agg["floor_price"] = agg["placement_group"].map(FLOOR_PRICES)
    agg["r_star"] = [
        optimal_hr_map[(pg, ck)]
        for pg, ck in zip(agg["placement_group"], agg["cohort_key"])
    ]
    agg["new_hr_applied"] = agg["r_star"]
    agg["revenue_lift"] = agg["ad_fee_after"] - agg["ad_fee_before"]
    agg["revenue_lift_pct"] = (
        (agg["ad_fee_after"] - agg["ad_fee_before"]) / agg["ad_fee_before"] * 100
    )
    agg["avg_cpc_before"] = agg["ad_fee_before"] / agg["n_rows"]
    agg["avg_cpc_after"] = agg["ad_fee_after"] / agg["n_rows"]

    result = agg.sort_values("revenue_lift", ascending=False).reset_index(drop=True)
    n_rows = len(result)
    for i, (_, row) in enumerate(result.iterrows(), 1):
        ck_label = _display_cohort_key(row['placement_group'], row['cohort_key'])
        print(
            f"  [{row['placement_group']} / {ck_label}  ({row['cohort_key']})]  "
            f"new_hr=${row['new_hr_applied']:.4f}  "
            f"before=${row['ad_fee_before']:.2f}  after=${row['ad_fee_after']:.2f}  "
            f"lift={row['revenue_lift_pct']:+.4f}%"
        )

    return result


# ── ROAS computation ──────────────────────────────────────────────────────────
def compute_roas(
    summary: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach ROAS before/after columns to summary using per-auction attributed
    sales.  Relies on capped_baseline / capped_new columns already present in
    eval_df (written by _apply_budget_caps).

    An auction's sales are counted only if its budget-capped CPC > 0 (i.e. the
    auction was funded before the campaign exhausted its daily budget).
    """
    sales = eval_df["attributed_sales_usd"].fillna(0.0)

    roas_df = eval_df.assign(
        sales_baseline=np.where(eval_df["capped_baseline"] > 0, sales, 0.0),
        sales_new=np.where(eval_df["capped_new"] > 0, sales, 0.0),
    )

    agg = (
        roas_df.groupby(["placement_group", "cohort_key"])
        .agg(
            spend_before=("capped_baseline", "sum"),
            spend_after=("capped_new", "sum"),
            sales_before=("sales_baseline", "sum"),
            sales_after=("sales_new", "sum"),
        )
        .reset_index()
    )

    agg["roas_before"] = np.where(
        agg["spend_before"] > 0,
        (agg["sales_before"] / agg["spend_before"]).round(4),
        0.0,
    )
    agg["roas_after"] = np.where(
        agg["spend_after"] > 0,
        (agg["sales_after"] / agg["spend_after"]).round(4),
        0.0,
    )
    agg["roas_change"] = (agg["roas_after"] - agg["roas_before"]).round(4)

    return summary.merge(
        agg[["placement_group", "cohort_key", "roas_before", "roas_after", "roas_change"]],
        on=["placement_group", "cohort_key"],
        how="left",
    )