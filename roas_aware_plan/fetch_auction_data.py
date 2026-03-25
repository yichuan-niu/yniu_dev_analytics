import json
from datetime import date, timedelta

import os
import pandas as pd
import snowflake.connector
#%%
# ── Constants ──────────────────────────────────────────────────────────────────
auction_date = "2026-03-21"
target_campaign_id = "0b76b55d-a017-4f77-a9a7-38fc41c90d2d"


DATABASE = "iguazu"
SCHEMA = "server_events_production"
TABLE = "auction_candidates_event_ice"

#%%
# ── Snowflake connection ───────────────────────────────────────────────────────

def get_connection() -> snowflake.connector.SnowflakeConnection:
    """Build a Snowflake connection from environment variables."""

    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"], 
        warehouse="ADHOC",   
        role=os.environ["SNOWFLAKE_ROLE"], 
        database=DATABASE,     
        schema=SCHEMA,   
    )



# ── Query ──────────────────────────────────────────────────────────────────────

QUERY = """
  WITH flattened AS (
      SELECT                                                                                                            
          AUCTION_ID,
          OCCURRED_AT,
          IGUAZU_PARTITION_DATE,
          IGUAZU_PARTITION_HOUR,
          c.value AS candidate
      FROM {database}.{schema}.{table},
      LATERAL FLATTEN(input => PARSE_JSON(CANDIDATES):candidates) c
      WHERE IGUAZU_PARTITION_DATE = '{partition_date}'
        AND CANDIDATES LIKE '%{campaign_id}%'
        AND (
            c.value:campaignId::STRING = '{campaign_id}'
            OR (c.value:auctionRank::INT = 0 AND c.value:campaignId::STRING != '{campaign_id}')
        )
  )
  SELECT
      AUCTION_ID,
      TO_JSON(
          OBJECT_CONSTRUCT('candidates', ARRAY_AGG(candidate) WITHIN GROUP (ORDER BY candidate:auctionRank::INT))
      ) AS CANDIDATES,
      ANY_VALUE(OCCURRED_AT)          AS OCCURRED_AT,
      ANY_VALUE(IGUAZU_PARTITION_DATE) AS IGUAZU_PARTITION_DATE,
      ANY_VALUE(IGUAZU_PARTITION_HOUR) AS IGUAZU_PARTITION_HOUR
  FROM flattened
  GROUP BY AUCTION_ID

"""


def fetch_auctions(partition_date: str, campaign_id: str) -> pd.DataFrame:
    """
    Fetch all auctions on `partition_date` where `campaign_id` participated.

    The LIKE filter on CANDIDATES is a fast pre-filter; we do a precise
    JSON check in Python afterward to ensure the campaign_id is actually
    a campaignId value (not a coincidental substring match elsewhere).

    Args:
        partition_date: 'YYYY-MM-DD' string
        campaign_id:    UUID of the campaign to look up

    Returns:
        DataFrame with one row per auction; CANDIDATES column is a parsed
        list of dicts (one dict per auction candidate).
    """
    query = QUERY.format(
        database=DATABASE, 
        schema=SCHEMA, 
        table=TABLE, 
        partition_date=partition_date, 
        campaign_id=campaign_id
    )

    print(f"Running query:\n{query}")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()

    print(f"  rows returned by SQL (pre-filter): {len(rows):,}")

    df = pd.DataFrame(rows, columns=columns)

    # Parse CANDIDATES JSON string → list of dicts
    df["candidates"] = df["candidates"].apply(parse_candidates)

    # Precise filter: keep only auctions where the campaign actually appears
    # as a campaignId value (guards against substring false positives)
    mask = df["candidates"].apply(
        lambda cands: any(c.get("campaignId") == campaign_id for c in cands)
    )
    df = df[mask].reset_index(drop=True)

    print(f"  auctions after precise filter:     {len(df):,}")
    return df


def parse_candidates(raw: str) -> list:
    """Parse CANDIDATES JSON string into a list of candidate dicts.
    Also parses the nested pricingMetadata string inside each candidate."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        candidates = data.get("candidates", [])
        for c in candidates:
            if isinstance(c.get("pricingMetadata"), str):
                try:
                    c["pricingMetadata"] = json.loads(c["pricingMetadata"])
                except json.JSONDecodeError:
                    pass
        return candidates
    except json.JSONDecodeError:
        return []


#%%
df = fetch_auctions(partition_date=auction_date, campaign_id=target_campaign_id)

print(f"\nDataFrame shape: {df.shape}")

df.to_pickle(f"data/auction_history_cmp_{target_campaign_id}.pkl")

