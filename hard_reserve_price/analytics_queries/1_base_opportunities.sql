-- Q01: Baseline opportunity rate
-- Condition: winner pays exactly hard_reserve AND hard_reserve >= GREATEST(raw_gsp, soft_reserve)
--   AND auction_bid > hard_reserve (headroom to raise the floor)
-- Interpretation: hard reserve is the sole binding floor (beats both GSP and soft reserve),
-- winner can afford to pay more — raising hard_reserve increases CPC directly.
-- Restricted to competitive auctions (final_auction_size > 1).
-- soft_reserve = softReserveBeta * nextBid

WITH winners AS (
    SELECT
        auction_id,
        auction_bid / 100.0                                                     AS auction_bid_dollars,
        bid_price_unit_amount / 100.0                                           AS cpc_dollars,
        GET(PARSE_JSON(pricing_metadata), 'cpcGsp')::INT / 100.0               AS raw_gsp_dollars,
        GET(PARSE_JSON(pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'softReserveBeta')::FLOAT
            * GET(PARSE_JSON(pricing_metadata), 'nextBid')::INT / 100.0        AS soft_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'finalAuctionSize')::INT              AS final_auction_size
    FROM edw.ads.ads_auction_candidates_event_delta SAMPLE (1)
    WHERE event_date = '2026-03-25'
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND GET(PARSE_JSON(pricing_metadata), 'finalAuctionSize')::INT > 1
)

SELECT
    COUNT(*)                                                                        AS total_auctions,
    SUM(CASE WHEN auction_bid_dollars > hard_reserve_dollars
              AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
              AND cpc_dollars = hard_reserve_dollars THEN 1 ELSE 0 END)            AS opportunity_auctions,
    ROUND(
        100.0 * SUM(CASE WHEN auction_bid_dollars > hard_reserve_dollars
                          AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                          AND cpc_dollars = hard_reserve_dollars THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                               AS opportunity_pct
FROM winners;
