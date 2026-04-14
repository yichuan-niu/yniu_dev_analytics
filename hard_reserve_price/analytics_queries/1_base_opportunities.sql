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
    FROM edw.ads.ads_auction_candidates_event_delta
    WHERE event_date = '2026-03-25'
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(auction_id)), 100) < 1
)

SELECT
    COUNT(*)                                                                        AS total_auctions,
    SUM(CASE WHEN final_auction_size > 1
              AND auction_bid_dollars > hard_reserve_dollars
              AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
              AND cpc_dollars = hard_reserve_dollars THEN 1 ELSE 0 END)            AS opportunity_auctions,
    ROUND(
        100.0 * SUM(CASE WHEN final_auction_size > 1
                          AND auction_bid_dollars > hard_reserve_dollars
                          AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                          AND cpc_dollars = hard_reserve_dollars THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                               AS opportunity_pct,
    ROUND(
        100.0 * SUM(CASE WHEN final_auction_size > 1
                          AND auction_bid_dollars > hard_reserve_dollars
                          AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                          AND cpc_dollars = hard_reserve_dollars THEN cpc_dollars ELSE NULL END)
        / NULLIF(SUM(CASE WHEN final_auction_size > 1
                           AND auction_bid_dollars > hard_reserve_dollars
                           AND hard_reserve_dollars >= GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                           AND cpc_dollars = hard_reserve_dollars THEN auction_bid_dollars ELSE NULL END), 0), 2
    )                                                                               AS cpc_to_bid_pct
FROM winners;


-- Q02: Case 2 — GSP/soft reserve is the binding floor (above hard reserve)
-- Condition: auction_bid > GREATEST(raw_gsp, soft_reserve) > hard_reserve
--   AND cpc = GREATEST(raw_gsp, soft_reserve) (winner pays the competitive floor)
-- Interpretation: natural competition sets the price, hard reserve is not binding.
-- Opportunity: raising hard_reserve above GREATEST(raw_gsp, soft_reserve) would
-- increase CPC — but only if hard_reserve is raised past the current competitive floor.
-- Restricted to competitive auctions (final_auction_size > 1).

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
    FROM edw.ads.ads_auction_candidates_event_delta
    WHERE event_date = '2026-03-25'
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(auction_id)), 100) < 1
)

SELECT
    COUNT(*)                                                                        AS total_auctions,
    SUM(CASE WHEN final_auction_size > 1
              AND auction_bid_dollars > GREATEST(raw_gsp_dollars, soft_reserve_dollars)
              AND GREATEST(raw_gsp_dollars, soft_reserve_dollars) > hard_reserve_dollars
              AND cpc_dollars = GREATEST(raw_gsp_dollars, soft_reserve_dollars) THEN 1 ELSE 0 END) AS opportunity_auctions,
    ROUND(
        100.0 * SUM(CASE WHEN final_auction_size > 1
                          AND auction_bid_dollars > GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                          AND GREATEST(raw_gsp_dollars, soft_reserve_dollars) > hard_reserve_dollars
                          AND cpc_dollars = GREATEST(raw_gsp_dollars, soft_reserve_dollars) THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                               AS opportunity_pct,
    ROUND(
        100.0 * SUM(CASE WHEN final_auction_size > 1
                          AND auction_bid_dollars > GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                          AND GREATEST(raw_gsp_dollars, soft_reserve_dollars) > hard_reserve_dollars
                          AND cpc_dollars = GREATEST(raw_gsp_dollars, soft_reserve_dollars) THEN cpc_dollars ELSE NULL END)
        / NULLIF(SUM(CASE WHEN final_auction_size > 1
                           AND auction_bid_dollars > GREATEST(raw_gsp_dollars, soft_reserve_dollars)
                           AND GREATEST(raw_gsp_dollars, soft_reserve_dollars) > hard_reserve_dollars
                           AND cpc_dollars = GREATEST(raw_gsp_dollars, soft_reserve_dollars) THEN auction_bid_dollars ELSE NULL END), 0), 2
    )                                                                               AS cpc_to_bid_pct
FROM winners;


-- Q03: Case 3 — Single-bidder auction, hard reserve is the sole price determinant
-- Condition: finalAuctionSize = 1 (no competitors, GSP/soft reserve do not apply)
--   AND cpc_dollars = hard_reserve_dollars (winner pays the hard reserve floor)
--   AND auction_bid_dollars > hard_reserve_dollars (headroom to raise the floor)
-- Interpretation: winner has no competitor, so hard reserve is the only binding
-- price floor. Raising hard reserve directly increases CPC up to the winner's bid.

WITH winners AS (
    SELECT
        auction_id,
        auction_bid / 100.0                                                     AS auction_bid_dollars,
        bid_price_unit_amount / 100.0                                           AS cpc_dollars,
        GET(PARSE_JSON(pricing_metadata), 'hardReserve')::INT / 100.0          AS hard_reserve_dollars,
        GET(PARSE_JSON(pricing_metadata), 'finalAuctionSize')::INT              AS final_auction_size
    FROM edw.ads.ads_auction_candidates_event_delta
    WHERE event_date = '2026-03-25'
      AND placement LIKE '%SPONSORED_PRODUCTS%'
      AND auction_rank = 0
      AND pricing_metadata IS NOT NULL
      AND MOD(ABS(HASH(auction_id)), 100) < 1
)

SELECT
    COUNT(*)                                                                        AS total_auctions,
    SUM(CASE WHEN final_auction_size = 1
              AND auction_bid_dollars > hard_reserve_dollars
              AND cpc_dollars = hard_reserve_dollars THEN 1 ELSE 0 END)            AS opportunity_auctions,
    ROUND(
        100.0 * SUM(CASE WHEN final_auction_size = 1
                          AND auction_bid_dollars > hard_reserve_dollars
                          AND cpc_dollars = hard_reserve_dollars THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                               AS opportunity_pct,
    ROUND(
        100.0 * SUM(CASE WHEN final_auction_size = 1
                          AND auction_bid_dollars > hard_reserve_dollars
                          AND cpc_dollars = hard_reserve_dollars THEN cpc_dollars ELSE NULL END)
        / NULLIF(SUM(CASE WHEN final_auction_size = 1
                           AND auction_bid_dollars > hard_reserve_dollars
                           AND cpc_dollars = hard_reserve_dollars THEN auction_bid_dollars ELSE NULL END), 0), 2
    )                                                                               AS cpc_to_bid_pct
FROM winners;
