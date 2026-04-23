import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import gamma as gamma_dist, lognorm

from simulation_customized_ctx_config import (
    PLACEMENT_TO_GROUP,
    PLACEMENT_GROUP_ORDER,
    COHORT_DIM,
    FLOOR_PRICES,
    get_connection,
)

plt.close("all")

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_START_DATE    = "2026-03-25"   # training window start (inclusive)
TRAIN_END_DATE      = "2026-03-31"   # training window end (inclusive)
EVAL_START_DATE     = "2026-04-01"   # evaluation window start (inclusive)
EVAL_END_DATE       = "2026-04-03"   # evaluation window end (inclusive)
TRAIN_SAMPLE_PCT    = 5              # auction-level sampling for training (MOD HASH < TRAIN_SAMPLE_PCT)
EVAL_SAMPLE_PCT     = 100            # campaign-level sampling for eval (100 = no sampling)
MAX_RANK            = 5              # use auction_rank < MAX_RANK for training bids
MIN_COHORT_BIDS     = 100            # min bid rows per cohort to fit a distribution
DIST_TYPE           = "gamma"        # "gamma" or "lognormal"
LOGNORM_SIGMA_MAX   = 1.2            # max sigma for lognormal (ensures monotone virtual valuation)
SELLER_VALUE        = 0.0            # Myerson seller valuation (v_0), usually 0
MAX_RESERVE_INC     = 5.0            # max allowed r* above floor (caps extreme tail fits)


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
  AND MOD(ABS(HASH(acd.auction_id)), 100) < {train_sample_pct}
"""

# ── Evaluation SQL ────────────────────────────────────────────────────────────
# All candidates (rank < max_rank) for clicked auctions over the eval date
# range.  Auction selection uses campaign-level sampling on the rank-0 winner,
# but once an auction is selected ALL its candidates are included (regardless
# of their campaign hash) so that runner-up promotion works correctly.
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
sampled_clicked_auctions AS (
    SELECT acd.auction_id
    FROM edw.ads.ads_auction_candidates_event_delta acd
    INNER JOIN clicked ON acd.auction_id = clicked.ad_auction_id
    WHERE acd.event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
      AND acd.CURRENCY_ISO_TYPE IN ('USD')
      AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
      AND acd.auction_rank = 0
      AND acd.pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(acd.campaign_id)), 100) < {eval_sample_pct}
)
SELECT
    acd.auction_id,
    acd.campaign_id,
    acd.placement,
    acd.event_date,
    acd.auction_rank,
    acd.auction_bid / 100.0                                                     AS auction_bid_dollars,
    acd.ad_score / 100.0                                                        AS ad_score_dollars,
    GET(PARSE_JSON(acd.pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
    GET(PARSE_JSON(acd.pricing_metadata), 'softReserveBeta')::FLOAT             AS soft_reserve_beta,
    COALESCE(acd.normalized_query, 'Unknown')                                  AS normalized_query,
    COALESCE(acd.l1_category_id::VARCHAR, 'Unknown')                           AS l1_category_id,
    COALESCE(acd.collection_id, 'Unknown')                                     AS collection_id,
    acd.event_hour                                                              AS hour_bucket,
    acd.event_timestamp                                                         AS auction_timestamp
FROM edw.ads.ads_auction_candidates_event_delta acd
WHERE acd.event_date BETWEEN '{eval_start_date}' AND '{eval_end_date}'
  AND acd.CURRENCY_ISO_TYPE IN ('USD')
  AND acd.placement LIKE '%SPONSORED_PRODUCTS%'
  AND acd.auction_rank < {max_rank}
  AND acd.pricing_metadata IS NOT NULL
  AND acd.auction_id IN (SELECT auction_id FROM sampled_clicked_auctions)
"""

BUDGET_QUERY = """
SELECT
    date_est,
    campaign_id,
    COALESCE(MAX(campaign_budget), SUM(daily_budget)) / 100 AS campaign_daily_budget_dollars
FROM PRODDB.PUBLIC.FACT_ADS_DAILY_BUDGET
WHERE date_est BETWEEN '{start_date}' AND '{end_date}'
GROUP BY date_est, campaign_id
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
    if not result.success:
        raise RuntimeError(f"Gamma MLE failed to converge: {result.message}")
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


def fit_lognormal_truncated(
    bids: np.ndarray, floor: float, sigma_max: float = LOGNORM_SIGMA_MAX,
):
    """Fit a Lognormal distribution to bids truncated from below at floor via L-BFGS-B.

    sigma is bounded by sigma_max to ensure the virtual valuation stays monotone
    (required for a unique Myerson optimal reserve).
    """
    log_bids = np.log(bids)
    result = minimize(
        _lognorm_nll,
        x0=[log_bids.mean(), min(log_bids.std() if log_bids.std() > 0 else 0.5, sigma_max)],
        args=(bids, floor),
        method="L-BFGS-B",
        bounds=[(None, None), (1e-6, sigma_max)],
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
    hi: float = 50.0,
    max_increment: float = MAX_RESERVE_INC,
):
    """
    Find r* where ψ(r*) = seller_value using Brent's bisection method.

    Returns the optimal reserve price if r* > floor, else None
    (meaning the theoretically optimal reserve is at or below the current floor,
    so no change is needed).  r* is capped at floor + max_increment to prevent
    extreme reserves from heavy-tailed fits.
    """
    vv = lambda v: virtual_valuation(v, dist) - seller_value
    # If virtual valuation at just above the floor is already >= 0,
    # the optimal reserve <= floor — no improvement possible.
    try:
        if vv(floor + 1e-6) >= 0:
            return None
        r_star = brentq(vv, floor + 1e-6, hi, xtol=1e-4)
    except ValueError:
        return None
    if r_star <= floor:
        return None
    return min(r_star, floor + max_increment)


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
        cohort_rows = df.loc[mask].dropna(subset=["auction_bid_dollars", "hard_reserve_dollars"])
        # Use per-row hard reserve for filtering (handles non-standard HR
        # from experiments or DV overrides within the training window)
        bids = cohort_rows.loc[
            cohort_rows["auction_bid_dollars"] > cohort_rows["hard_reserve_dollars"],
            "auction_bid_dollars",
        ].to_numpy(dtype=float)

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
        budgets_bl = day["campaign_id"].map(
            lambda c, b=day_budget: b.get(c, float("inf"))
        )
        cum_bl = day.groupby("campaign_id")["cpc_baseline"].cumsum()
        remaining_bl = (budgets_bl - (cum_bl - day["cpc_baseline"])).clip(lower=0)
        df.loc[day.index, "capped_baseline"] = (
            day["cpc_baseline"].clip(upper=remaining_bl).values
        )

        # ── New: budget tracked by new_campaign_id ────────────────────────
        budgets_nw = day["new_campaign_id"].map(
            lambda c, b=day_budget: b.get(c, float("inf"))
        )
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
    agg["revenue_lift_pct"] = (
        (agg["ad_fee_after"] - agg["ad_fee_before"]) / agg["ad_fee_before"] * 100
    )
    agg["avg_cpc_before"] = agg["ad_fee_before"] / agg["n_rows"]
    agg["avg_cpc_after"] = agg["ad_fee_after"] / agg["n_rows"]

    result = agg.sort_values("revenue_lift_pct", ascending=False).reset_index(drop=True)
    for _, row in result.iterrows():
        print(
            f"  [{row['placement_group']} / {row['cohort_key']}]  "
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

#%% Evaluation: fetch all candidates for clicked auctions
print(f"\nFetching evaluation data ({EVAL_START_DATE} – {EVAL_END_DATE})...")

eval_all = fetch_eval_data()
eval_all.to_pickle(f"data/simulation_ctx_eval_{EVAL_START_DATE}_to_{EVAL_END_DATE}_smpl_{EVAL_SAMPLE_PCT}_df.pkl")

# eval_all = pd.read_pickle("data/simulation_ctx_eval_df.pkl")

print(f"  Eval candidate rows: {len(eval_all):,}")
print(f"  Unique auctions:     {eval_all['auction_id'].nunique():,}")
rank0_only = eval_all[eval_all["auction_rank"] == 0]
print(f"  Eval total CPC ($):  (derived from formula after resolve_auction_outcomes)")

#%% Fetch budget and ROAS data
print(f"\nFetching campaign daily budgets ({EVAL_START_DATE} – {EVAL_END_DATE})...")

budget_df = fetch_budget()
budget_df.to_pickle(f"data/simulation_ctx_budget_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

# budget_df = pd.read_pickle(f"data/simulation_ctx_budget_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

budget_maps = {
    dt: grp.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
    for dt, grp in budget_df.groupby("date_est")
}
print(f"  Budget dates: {sorted(budget_maps.keys())}")
print(f"  Campaigns with budget (total): {budget_df['campaign_id'].nunique():,}")

print(f"\nFetching per-auction attributed sales ({EVAL_START_DATE} – {EVAL_END_DATE})...")

sales_df = fetch_sales()
sales_df.to_pickle(f"data/simulation_ctx_sales_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

# sales_df = pd.read_pickle(f"data/simulation_ctx_sales_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

print(f"  Auctions with sales: {len(sales_df):,}")

#%% Run evaluation replay
print("\nRunning auction replay with Myerson-optimal reserves...")
# Add cohort columns to ALL candidates (needed for new_hr lookup in resolve)
eval_all = _add_cohort_columns(eval_all)

# Resolve per-auction winners with runner-up promotion, then join sales
eval_df = resolve_auction_outcomes(eval_all, optimal_hr_map)
eval_df = eval_df.merge(sales_df, on="auction_id", how="left")

_apply_budget_caps(eval_df, budget_maps)

summary = evaluate_all_cohorts(eval_df, optimal_hr_map)

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
# Monetization rate = sum(CPC) / sum(bid)
# Uses capped_baseline / capped_new from _apply_budget_caps for consistency
# with the revenue lift and ROAS calculations (same formula, same budget caps).

bid = eval_df["auction_bid_dollars"]
new_bid = eval_df["new_bid"]

# ── Global monetization rate ──────────────────────────────────────────────────
total_bid = bid.sum()
total_new_bid = new_bid.sum()
mr_before_global = eval_df["capped_baseline"].sum() / total_bid if total_bid > 0 else np.nan
mr_after_global  = eval_df["capped_new"].sum() / total_new_bid if total_new_bid > 0 else np.nan

print(f"\n{'─' * 55}")
print(f"{'Monetization Rate (MR = CPC / bid)':^55}")
print(f"{'─' * 55}")
print(f"  Global MR before: {mr_before_global:.4%}")
print(f"  Global MR after:  {mr_after_global:.4%}")
print(f"  Global MR delta:  {mr_after_global - mr_before_global:+.4%}")
print(f"{'─' * 55}")

# ── Per-cohort monetization rate ──────────────────────────────────────────────
cohort_mr = (
    eval_df.groupby(["placement_group", "cohort_key"])
    .agg(
        bid_sum=("auction_bid_dollars", "sum"),
        new_bid_sum=("new_bid", "sum"),
        cpc_sum=("capped_baseline", "sum"),
        new_cpc_sum=("capped_new", "sum"),
    )
    .reset_index()
)
cohort_mr["mr_before"] = np.where(
    cohort_mr["bid_sum"] > 0,
    cohort_mr["cpc_sum"] / cohort_mr["bid_sum"],
    0.0,
)
cohort_mr["mr_after"]  = np.where(
    cohort_mr["new_bid_sum"] > 0,
    cohort_mr["new_cpc_sum"] / cohort_mr["new_bid_sum"],
    0.0,
)
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
