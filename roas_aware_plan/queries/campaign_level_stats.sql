WITH conversion_rows AS (
  SELECT
      snapshot_date_utc,
      product_type,
      touchpoint_dd_sic,
      conversion_event_timestamp_utc
  FROM EDW.ADS_DATA.FACT_AD_EVENT_ATTRIBUTION
  WHERE touchpoint_dd_sic in (
        SELECT DISTINCT
            ad_entity_id AS dd_sic
        FROM
            edw.ads.dim_ad_entity_history
        WHERE
            campaign_id = '0b76b55d-a017-4f77-a9a7-38fc41c90d2d'
            AND ad_entity_id_type = 'DD_SIC'
            AND valid_to_utc IS NULL
  )
    AND attribution_model = 'last_touch__first_conversion'
    AND conversion_event_type = 'order_cart_submission'
    and touchpoint_event_type = 'impression'
    and product_type IN ('organic_cpg', 'sp')
    AND snapshot_date_utc BETWEEN '2026-03-10' AND '2026-03-20'
),
converison_table as (
    SELECT
        snapshot_date_utc,
        CASE
          WHEN product_type = 'organic_cpg' THEN 'organic'
          ELSE 'ad'
        END AS conversion_source,
        COUNT(*) AS conversions
    FROM conversion_rows
    GROUP BY 1, 2
),

impression_table as (
    SELECT
        snapshot_date_utc,
        CASE
          WHEN event_product_type = 'organic_cpg' THEN 'organic'
          ELSE 'ad'
        END AS conversion_source,

        SUM(CASE WHEN event_type = 'impression' THEN 1 ELSE 0 END)  AS impressions,
    FROM edw.ads_data.fact_ads_unified_transaction_event
    WHERE snapshot_date_utc BETWEEN '2026-03-10' AND '2026-03-20'
      AND event_product_type IN ('organic_cpg', 'sp')
      AND dd_sic_v1 in (

      SELECT DISTINCT
        ad_entity_id AS dd_sic
    FROM edw.ads.dim_ad_entity_history
    WHERE campaign_id = '0b76b55d-a017-4f77-a9a7-38fc41c90d2d'
      AND ad_entity_id_type = 'DD_SIC'
      AND valid_to_utc IS NULL

      )
      and event_deduped_flag       -- exclude duplicates
    group by 1, 2
)

select
    conversion_source,
    snapshot_date_utc as serve_date_utc,
    i.impressions,
    c.conversions,
    ifnull(c.conversions, 0) / i.impressions  * 100 as cvr
from
    impression_table as i
left join
    converison_table as c
using
    (snapshot_date_utc, conversion_source)
order by 1, 2
;

