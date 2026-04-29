"""Shared constants and helpers for hard reserve price analysis scripts."""

import os
import pandas as pd
import snowflake.connector


# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_START_DATE    = "2026-03-31"   # training window start (inclusive)
TRAIN_END_DATE      = "2026-03-31"   # training window end (inclusive)
EVAL_START_DATE     = "2026-04-01"   # evaluation window start (inclusive)
EVAL_END_DATE       = "2026-04-03"   # evaluation window end (inclusive)
TRAIN_SAMPLE_PCT    = 1           # auction-level sampling for training (MOD HASH(1M) < PCT; 10000 = 1%)
EVAL_SAMPLE_PCT     = 50            # campaign-level sampling for eval (100 = no sampling)
MAX_RANK            = 20              # use auction_rank < MAX_RANK for training bids
MIN_COHORT_BIDS     = 100           # min bid rows per cohort to fit a distribution
TOP_N_COHORTS       = 10000           # per placement group: keep only top-N cohorts by bid count (None = no limit)
DIST_TYPE           = "lognormal"        # "gamma" or "lognormal"
LOGNORM_SIGMA_MAX   = 1.2            # max sigma for lognormal (ensures monotone virtual valuation)
SELLER_VALUE        = 0.0            # Myerson seller valuation (v_0), usually 0
MAX_RESERVE_INC     = 5.0            # absolute cap on final hard reserve (caps extreme tail fits)
KEEP_RESERVE_BELOW_FLOOR = False     # True = keep r* <= floor for eval; False = discard them
# Per-placement-group scaler applied to the full reserve:
#   new_hr = r* * scaler   (then capped at MAX_RESERVE_INC)
# 1.0 = use Myerson r* unchanged; <1.0 = more conservative; >1.0 = more aggressive.
AGGRESSIVENESS_SCALER = {
    "Search":     1.0,
    "Category":   1.0,
    "Collection": 0.1,
    "DoubleDash": 0.1,
}

# Category ID → name mapping sourced from CATALOG_SERVICE_PROD.public.PRODUCT_CATEGORY.
# To refresh: re-query and save with pickle.dump({str(r['id']): r['name'] for r in rows}, open(path, 'wb')).
L1_CATEGORY_NAMES: dict = pd.read_pickle("../data/l1_category_names.pkl")


# ── Placement groups ─────────────────────────────────────────────────────────
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
PLACEMENT_GROUP_ORDER = ["Search", "Category", "Collection", "DoubleDash"]

# Secondary cohort dimension per placement group.
COHORT_DIM = {
    "Search":     "normalized_query",
    "Category":   "l1_category_id",
    "Collection": "collection_id",
    "DoubleDash": "hour_bucket",
}

# Default hard reserve / floor price per placement group (USD).
FLOOR_PRICES = {
    "Search":     0.60,
    "Category":   0.40,
    "Collection": 0.30,
    "DoubleDash": 0.80,
}


# ── Snowflake connection ────────────────────────────────────────────────────
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
