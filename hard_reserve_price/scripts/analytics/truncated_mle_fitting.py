import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
plt.close("all")


def truncated_lognorm_nll(params, bids, floor_price):
    """Negative log-likelihood for a lognormal truncated from below at floor_price."""
    mu, sigma = params
    dist = lognorm(s=sigma, scale=np.exp(mu))
    log_lik = np.sum(dist.logpdf(bids)) - len(bids) * np.log(1 - dist.cdf(floor_price))
    return -log_lik


def virtual_valuation(v, dist):
    """Myerson virtual valuation: psi(v) = v - (1 - F(v)) / f(v)"""
    return v - (1 - dist.cdf(v)) / dist.pdf(v)


# ── 1. Data ───────────────────────────────────────────────────────────────────
observed_bids = np.array([1.25, 1.40, 1.80, 2.10, 1.35])
current_floor = 1.2   # existing floor price used as truncation point
seller_value  = 0.0   # seller's own valuation (v_0), often 0

# ── 2. Truncated MLE ──────────────────────────────────────────────────────────
initial_guess = [np.mean(np.log(observed_bids)), np.std(np.log(observed_bids))]

result = minimize(
    truncated_lognorm_nll,
    initial_guess,
    args=(observed_bids, current_floor),
    method="L-BFGS-B",
    bounds=[(None, None), (1e-6, None)],
)

if not result.success:
    raise RuntimeError(f"Optimization failed: {result.message}")

est_mu, est_sigma = result.x

dist_fit = lognorm(s=est_sigma, scale=np.exp(est_mu))

print(f"Estimated mu:    {est_mu:.4f}")
print(f"Estimated sigma: {est_sigma:.4f}")
print(f"Implied lognormal mean: {np.exp(est_mu + est_sigma**2 / 2):.4f}")

# ── 3. Optimal Reserve Price (Myerson) ───────────────────────────────────────
# Solve psi(r*) = seller_value, i.e. r* - (1 - F(r*)) / f(r*) = seller_value
# Bracket: psi is increasing; find [lo, hi] where it crosses seller_value
lo, hi = 1e-6, 100.0
optimal_reserve = brentq(lambda v: virtual_valuation(v, dist_fit) - seller_value, lo, hi)

print(f"\nOptimal reserve price (Myerson): {optimal_reserve:.4f}")
print(f"Current floor price:             {current_floor:.4f}")

# ── 4. Plots ──────────────────────────────────────────────────────────────────
x = np.linspace(0.3, 3.5, 500)
trunc_pdf = np.where(
    x >= current_floor,
    dist_fit.pdf(x) / (1 - dist_fit.cdf(current_floor)),
    0,
)
vv = np.array([virtual_valuation(v, dist_fit) for v in x])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1: Full vs truncated PDF
ax = axes[0]
ax.plot(x, dist_fit.pdf(x), color="steelblue", lw=2, label="Full lognormal (fitted)")
ax.plot(x, trunc_pdf, color="darkorange", lw=2, linestyle="--", label="Truncated PDF (observed)")
ax.fill_between(x, dist_fit.pdf(x), where=(x < current_floor), alpha=0.15, color="red", label="Unobserved region")
ax.axvline(current_floor,   color="red",    linestyle=":",  lw=1.5, label=f"Current floor = {current_floor}")
ax.axvline(optimal_reserve, color="green",  linestyle="--", lw=1.5, label=f"Optimal reserve = {optimal_reserve:.2f}")
ax.scatter(observed_bids, np.zeros_like(observed_bids), color="black", marker="|", s=200, zorder=5, label="Observed bids")
ax.set_xlabel("Bid")
ax.set_ylabel("Density")
ax.set_title("Full vs Truncated Lognormal PDF")
ax.legend(fontsize=7)

# Plot 2: Histogram vs fitted truncated PDF
ax = axes[1]
ax.hist(observed_bids, bins=5, density=True, alpha=0.5, color="steelblue", edgecolor="white", label="Observed bids")
ax.plot(x[x >= current_floor], trunc_pdf[x >= current_floor], color="darkorange", lw=2, label="Fitted truncated PDF")
ax.axvline(current_floor,   color="red",   linestyle=":",  lw=1.5, label=f"Current floor = {current_floor}")
ax.axvline(optimal_reserve, color="green", linestyle="--", lw=1.5, label=f"Optimal reserve = {optimal_reserve:.2f}")
ax.set_xlabel("Bid")
ax.set_ylabel("Density")
ax.set_title("Histogram vs Fitted Truncated PDF")
ax.legend(fontsize=7)

# Plot 3: Virtual valuation curve
ax = axes[2]
ax.plot(x, vv, color="purple", lw=2, label="Virtual valuation ψ(v)")
ax.axhline(seller_value,    color="gray",  linestyle=":",  lw=1.5, label=f"Seller value v₀ = {seller_value}")
ax.axvline(optimal_reserve, color="green", linestyle="--", lw=1.5, label=f"Optimal reserve r* = {optimal_reserve:.2f}")
ax.scatter([optimal_reserve], [seller_value], color="green", zorder=5, s=80)
ax.set_xlabel("v")
ax.set_ylabel("ψ(v)")
ax.set_title("Myerson Virtual Valuation")
ax.legend(fontsize=7)
ax.set_ylim(-2, 3)

plt.tight_layout()
plt.show()

# ── 5. Monotonicity check ─────────────────────────────────────────────────────
x_full = np.linspace(0.5, 5, 1000)
vv_full = np.array([virtual_valuation(v, dist_fit) for v in x_full])
dvv = np.diff(vv_full) / np.diff(x_full)   # numerical derivative ψ'(v)
x_mid = (x_full[:-1] + x_full[1:]) / 2     # midpoints for derivative plot

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

# Left: ψ(v) over full [lo, hi]
ax = axes2[0]
ax.plot(x_full, vv_full, color="purple", lw=2, label="ψ(v)")
ax.axhline(seller_value,    color="gray",  linestyle=":", lw=1.5, label=f"v₀ = {seller_value}")
ax.axvline(optimal_reserve, color="green", linestyle="--", lw=1.5, label=f"r* = {optimal_reserve:.2f}")
ax.scatter([optimal_reserve], [seller_value], color="green", zorder=5, s=80)
ax.set_xlabel("v")
ax.set_ylabel("ψ(v)")
ax.set_title(f"Virtual Valuation over [{lo}, {hi}]")
ax.legend(fontsize=8)

# Right: ψ'(v) — monotonic iff always positive
ax = axes2[1]
ax.plot(x_mid, dvv, color="darkorange", lw=2, label="ψ'(v)")
ax.axhline(0, color="red", linestyle=":", lw=1.5, label="zero")
ax.fill_between(x_mid, dvv, 0, where=(dvv >= 0), alpha=0.15, color="green", label="positive (monotone)")
ax.fill_between(x_mid, dvv, 0, where=(dvv <  0), alpha=0.15, color="red",   label="negative (violation)")
ax.set_xlabel("v")
ax.set_ylabel("ψ'(v)")
ax.set_title("Derivative of Virtual Valuation")
ax.legend(fontsize=8)

print(f"\nMonotonicity check: min ψ'(v) = {dvv.min():.6f} → {'PASS' if dvv.min() > 0 else 'FAIL'}")

plt.tight_layout()
plt.show()
