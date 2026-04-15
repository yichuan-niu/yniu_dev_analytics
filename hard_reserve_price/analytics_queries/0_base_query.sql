SELECT
    auction_rank,
    ad_score / 100 as ad_score_dollars,
    -- true_bid / 100 as true_bid_dollars,
    auction_bid / 100 as auction_bid_dollars,
    bid_price_unit_amount / 100.0 AS cpc_dollars,
    GET(PARSE_JSON(PRICING_METADATA), 'cpcGsp')::INT / 100.0 AS raw_gsp_dollars,
    GET(PARSE_JSON(PRICING_METADATA), 'hardReserve')::INT / 100.0 AS hard_reserve_dollars,
    GET(PARSE_JSON(PRICING_METADATA), 'finalAuctionSize')::INT AS final_auction_size,
    normalized_query,
    currency_iso_type,
    collection_id,
    submarket_id,
    placement,
    dd_sic,
    auction_id,
FROM edw.ads.ads_auction_candidates_event_delta SAMPLE (1)
WHERE event_date = '2026-03-25'
  AND event_hour = 12
  and placement like '%SPONSORED_PRODUCTS%'
;



SELECT
    count(distinct auction_id)
FROM edw.ads.ads_auction_candidates_event_delta SAMPLE (1)
WHERE event_date = '2026-03-25'
  and placement like '%SPONSORED_PRODUCTS%'
;