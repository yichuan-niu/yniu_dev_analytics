import warnings
warnings.filterwarnings("ignore")
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation_customized_ctx_config import (
    PLACEMENT_GROUP_ORDER,
    COHORT_DIM,
    FLOOR_PRICES,
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
)

from simulation_customized_ctx_lib import (
    _display_cohort_key,
    fetch_train_data,
    fetch_eval_data,
    fetch_budget,
    fetch_sales,
    _add_cohort_columns,
    train_optimal_reserves,
    resolve_auction_outcomes,
    _apply_budget_caps,
    evaluate_all_cohorts,
    compute_roas,
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


def plot_bid_distribution(
    pg: str,
    ck,
    train_df: pd.DataFrame,
    eval_all: pd.DataFrame,
    fitted_dist=None,
    *,
    reserve_price=None,
) -> None:
    """Plot bid histograms with training-fitted PDF overlay for a single cohort.

    Overlays the same training-fitted truncated PDF on both train and eval
    histograms so you can assess (1) how well the parametric model captures
    the training data and (2) how well it generalises to the eval period.

    Parameters
    ----------
    fitted_dist : frozen scipy distribution, optional
        The distribution fitted on training bids during train_optimal_reserves().
        If None, the PDF overlay is skipped.
    """
    train = _add_cohort_columns(train_df)
    floor = FLOOR_PRICES[pg]

    train_bids = train.loc[
        (train["placement_group"] == pg) & (train["cohort_key"] == ck),
        "auction_bid_dollars",
    ]
    eval_bids = eval_all.loc[
        (eval_all["placement_group"] == pg) & (eval_all["cohort_key"] == ck),
        "auction_bid_dollars",
    ]

    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(14, 5))
    x_max = max(train_bids.quantile(0.99), eval_bids.quantile(0.99))
    bins = np.linspace(0, x_max, 60)

    # Truncated PDF from training fit (shared across both subplots)
    if fitted_dist is not None:
        x_pdf = np.linspace(floor, x_max, 300)
        surv = 1.0 - fitted_dist.cdf(floor)
        pdf_vals = fitted_dist.pdf(x_pdf) / surv if surv > 0 else fitted_dist.pdf(x_pdf)

    for ax, bids, label, color in [
        (ax_train, train_bids, "Train", "steelblue"),
        (ax_eval, eval_bids, "Eval", "darkorange"),
    ]:
        ax.hist(bids, bins=bins, density=True, color=color, edgecolor="white",
                alpha=0.7)

        # Overlay training-fitted PDF, scaled by fraction of bids above floor
        if fitted_dist is not None:
            frac_above = (bids > floor).sum() / len(bids)
            ax.plot(x_pdf, pdf_vals * frac_above, color="black", lw=1.5,
                    label=f"Train-fitted {DIST_TYPE}")

        if reserve_price is not None:
            ax.axvline(reserve_price, color="red", linestyle="--", lw=1.5,
                       label=f"Reserve=${reserve_price:.4f}")
        ax.axvline(floor, color="gray", linestyle=":", lw=1,
                   label=f"Floor=${floor:.2f}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Bid ($)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} (n={len(bids):,})")

    fig.suptitle(f"Bid Distribution: {pg} / {ck}", fontsize=13)
    plt.tight_layout()
    plt.show()


def debug_cohort(
    pg: str,
    ck,
    train_df: pd.DataFrame,
    eval_all: pd.DataFrame,
    budget_maps: dict,
    fitted_dist=None,
    *,
    reserve_price=None,
) -> None:
    """Inspect bid distributions and revenue for a single (placement_group, cohort_key).

    Re-runs the full auction replay with ONLY this cohort's reserve changed
    (all other cohorts keep their original hard reserve), then applies budget
    caps to compute the isolated revenue lift.
    """
    plot_bid_distribution(pg, ck, train_df, eval_all, fitted_dist,
                          reserve_price=reserve_price)

    if reserve_price is None:
        return

    # ── Re-run auction replay with only this cohort's reserve changed ────
    single_hr_map = {(pg, ck): reserve_price}
    eval_df = resolve_auction_outcomes(eval_all, single_hr_map)
    _apply_budget_caps(eval_df, budget_maps)

    # ── Revenue at given reserve ──────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  Reserve price: ${reserve_price:.4f}")
    print(f"{'─' * 55}")

    # Training: each bid >= reserve represents a won auction paying at least r
    train = _add_cohort_columns(train_df)
    train_bids = train.loc[
        (train["placement_group"] == pg) & (train["cohort_key"] == ck),
        "auction_bid_dollars",
    ]
    n_train = len(train_bids)
    n_above_train = (train_bids >= reserve_price).sum()
    pct_train = n_above_train / n_train * 100 if n_train > 0 else 0
    rev_train = n_above_train * reserve_price
    print(f"  Train bids above reserve: {n_above_train:,} / {n_train:,} ({pct_train:.1f}%)")
    print(f"  Train revenue lower bound: ${rev_train:,.2f}")

    # Eval: isolated replay results for this cohort
    cohort = eval_df[
        (eval_df["placement_group"] == pg) & (eval_df["cohort_key"] == ck)
    ]
    n_eval = len(cohort)
    n_above_eval = (cohort["auction_bid_dollars"] >= reserve_price).sum()
    n_lost = n_eval - n_above_eval
    rev_before = cohort["capped_baseline"].sum()
    rev_after = cohort["capped_new"].sum()
    lift = rev_after - rev_before
    lift_pct = lift / rev_before * 100 if rev_before > 0 else 0.0
    print(f"\n  Eval auctions: {n_eval:,}  (above reserve: {n_above_eval:,}, lost: {n_lost:,})")
    print(f"  Eval revenue before (capped):    ${rev_before:,.2f}")
    print(f"  Eval revenue after  (capped):    ${rev_after:,.2f}")
    print(f"  Eval lift:                       ${lift:,.2f}  ({lift_pct:+.4f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
# Training: fetch auction candidates from clicked auctions over training window

# train_df = fetch_train_data()
# train_df.to_pickle(f"../data/simulation_ctx_train_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}_smpl_{TRAIN_SAMPLE_PCT}_max_rank_{MAX_RANK}_df.pkl")

train_df = pd.read_pickle(f"../data/simulation_ctx_train_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}_smpl_{TRAIN_SAMPLE_PCT}_max_rank_{MAX_RANK}_df.pkl")

print(f"  Training rows: {len(train_df):,}")

# Evaluation: fetch all candidates for clicked auctions
print(f"\nFetching evaluation data ({EVAL_START_DATE} – {EVAL_END_DATE})...")

# eval_all = fetch_eval_data()
# eval_all.to_pickle(f"../data/simulation_ctx_eval_{EVAL_START_DATE}_to_{EVAL_END_DATE}_smpl_{EVAL_SAMPLE_PCT}_max_rank_{MAX_RANK}_df.pkl")

eval_all = pd.read_pickle(f"../data/simulation_ctx_eval_{EVAL_START_DATE}_to_{EVAL_END_DATE}_smpl_{EVAL_SAMPLE_PCT}_max_rank_{MAX_RANK}_df.pkl")

print(f"  Eval candidate rows: {len(eval_all):,}")
print(f"  Unique auctions:     {eval_all['auction_id'].nunique():,}")

# Fetch budget and ROAS data
print(f"\nFetching campaign daily budgets ({EVAL_START_DATE} – {EVAL_END_DATE})...")

# budget_df = fetch_budget()
# budget_df.to_pickle(f"../data/simulation_ctx_budget_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

budget_df = pd.read_pickle(f"../data/simulation_ctx_budget_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

budget_maps = {
    dt: grp.set_index("campaign_id")["campaign_daily_budget_dollars"].to_dict()
    for dt, grp in budget_df.groupby("date_est")
}
print(f"  Budget dates: {sorted(budget_maps.keys())}")
print(f"  Campaigns with budget (total): {budget_df['campaign_id'].nunique():,}")

print(f"\nFetching per-auction attributed sales ({EVAL_START_DATE} – {EVAL_END_DATE})...")

# sales_df = fetch_sales()
# sales_df.to_pickle(f"../data/simulation_ctx_sales_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

sales_df = pd.read_pickle(f"../data/simulation_ctx_sales_{EVAL_START_DATE}_to_{EVAL_END_DATE}_df.pkl")

print(f"  Auctions with sales: {len(sales_df):,}")

# Fit distributions and solve Myerson's equation per cohort
print(f"\n{'─' * 60}")
print("Simulation Parameters")
print(f"{'─' * 60}")
print(f"  TRAIN_START_DATE  = {TRAIN_START_DATE}")
print(f"  TRAIN_END_DATE    = {TRAIN_END_DATE}")
print(f"  EVAL_START_DATE   = {EVAL_START_DATE}")
print(f"  EVAL_END_DATE     = {EVAL_END_DATE}")
print(f"  TRAIN_SAMPLE_PCT  = {TRAIN_SAMPLE_PCT}")
print(f"  EVAL_SAMPLE_PCT   = {EVAL_SAMPLE_PCT}")
print(f"  MAX_RANK          = {MAX_RANK}")
print(f"  MIN_COHORT_BIDS   = {MIN_COHORT_BIDS:,}")
print(f"  TOP_N_COHORTS     = {TOP_N_COHORTS}")
print(f"  DIST_TYPE         = {DIST_TYPE}")
print(f"  LOGNORM_SIGMA_MAX = {LOGNORM_SIGMA_MAX}")
print(f"  SELLER_VALUE      = {SELLER_VALUE}")
print(f"  MAX_RESERVE_INC   = {MAX_RESERVE_INC}")
print(f"{'─' * 60}")

print("\nFitting distributions and solving for Myerson optimal reserves...")
optimal_hr_map, fitted_dists = train_optimal_reserves(
    train_df,
    min_cohort_bids=MIN_COHORT_BIDS,
    dist_type=DIST_TYPE,
    top_n_cohorts=TOP_N_COHORTS,
)

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

# Run evaluation replay
print("\nRunning auction replay with Myerson-optimal reserves...")
# Add cohort columns to ALL candidates (needed for new_hr lookup in resolve)
eval_all = _add_cohort_columns(eval_all)

# Resolve per-auction winners with runner-up promotion, then join sales
eval_df = resolve_auction_outcomes(eval_all, optimal_hr_map)
eval_df = eval_df.merge(sales_df, on="auction_id", how="left")

_apply_budget_caps(eval_df, budget_maps)

summary = evaluate_all_cohorts(eval_df, optimal_hr_map)

# Total revenue summary (all campaigns, eval dates)
rev_before = eval_df["capped_baseline"].sum()
rev_after = eval_df["capped_new"].sum()
lift_amt = rev_after - rev_before
lift_pct = lift_amt / rev_before * 100 if rev_before > 0 else 0.0
print(f"\n{'═' * 60}")
print("Total Revenue (all campaigns, eval dates)")
print(f"{'═' * 60}")
print(f"  Before:       ${rev_before:>14,.2f}")
print(f"  After:        ${rev_after:>14,.2f}")
print(f"  Lift:         ${lift_amt:>14,.2f}")
print(f"  Lift %:        {lift_pct:>14.4f}%")
print(f"{'═' * 60}")
#%%
plt.close("all")

debug_cohort("Collection", "recommended", train_df, eval_all, budget_maps,
             fitted_dists.get(("Collection", "recommended")), reserve_price=1.2926)

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
changed_cpc = summary["ad_fee_before"].sum()
total_sp_cpc = eval_df["capped_baseline"].sum()
lift_pct_changed = total_lift_dollars / changed_cpc * 100 if changed_cpc > 0 else 0.0
lift_pct_total = total_lift_dollars / total_sp_cpc * 100 if total_sp_cpc > 0 else 0.0
print(f"\n{'─' * 97}")
print(
    f"Lift on changed cohorts: {lift_pct_changed:.4f}%"
    f"  (${total_lift_dollars:,.2f} on ${changed_cpc:,.2f} CPC)"
)
print(
    f"Lift on total SP USD:    {lift_pct_total:.4f}%"
    f"  (${total_lift_dollars:,.2f} on ${total_sp_cpc:,.2f} CPC)"
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
