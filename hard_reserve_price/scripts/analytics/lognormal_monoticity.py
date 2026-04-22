"""
Regularity condition for lognormal virtual valuation
=====================================================

For lognormal(mu, sigma), the virtual valuation is:

    psi(v) = v - (1 - F(v)) / f(v)

Substituting z = (log v - mu) / sigma and using the Mills ratio R(z) = (1-Phi(z))/phi(z):

    psi(v) = v * (1 - sigma * R(z))

Taking the derivative (via chain rule, using R'(z) = z*R(z) - 1):

    psi'(v) = 2 - R(z) * (sigma + z)

So psi is monotonically increasing at v iff:

    R(z) * (sigma + z) < 2     ... (*)

Global monotonicity (all v > 0, i.e. z in (-inf, inf)) depends only on sigma.
mu is irrelevant globally because z sweeps all of (-inf, inf) regardless of mu.

If we restrict to v >= floor_price, z starts at z_floor = (log(floor) - mu) / sigma,
so mu re-enters: higher mu pushes z_floor lower (deeper into the safe region).
"""

import numpy as np
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt

plt.close("all")

# Mills ratio R(z) = (1 - Phi(z)) / phi(z)
def mills_ratio(z):
    return (1 - norm.cdf(z)) / norm.pdf(z)

# Condition (*): 2 - R(z)*(sigma + z) > 0
def regularity_margin(z, sigma):
    return 2 - mills_ratio(z) * (sigma + z)

# ── 1. Global check: does condition hold for ALL z, as a function of sigma? ───
sigmas = np.linspace(0.01, 3.0, 300)
z_grid = np.linspace(-10, 10, 5000)

# Minimum margin over all z, for each sigma
# If min > 0, psi is globally monotone
min_margin = np.array([regularity_margin(z_grid, s).min() for s in sigmas])

# Critical sigma: largest sigma where min_margin >= 0
critical_sigma = sigmas[np.where(min_margin >= 0)[0][-1]] if np.any(min_margin >= 0) else None

# ── 2. Effect of floor price: check monotonicity only for v >= floor_price ───
floor_price = 1.2
mu_values   = [0.0, 0.5, 1.0]
sigma_check = 1.5   # a sigma that may be globally non-monotone

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1: min margin vs sigma (global regularity)
ax = axes[0]
ax.plot(sigmas, min_margin, color="purple", lw=2)
ax.axhline(0, color="red", linestyle=":", lw=1.5, label="Threshold")
ax.fill_between(sigmas, min_margin, 0, where=(min_margin >= 0), alpha=0.15, color="green", label="Monotone (regular)")
ax.fill_between(sigmas, min_margin, 0, where=(min_margin <  0), alpha=0.15, color="red",   label="Non-monotone (irregular)")
if critical_sigma:
    ax.axvline(critical_sigma, color="black", linestyle="--", lw=1.5, label=f"σ* ≈ {critical_sigma:.2f}")
ax.set_xlabel("σ")
ax.set_ylabel("min  ψ'(v)  over all v > 0")
ax.set_title("Global regularity vs σ\n(μ does not matter)")
ax.legend(fontsize=7)

# Plot 2: regularity_margin(z, sigma) as a heatmap over (z, sigma)
ax = axes[1]
Z, S = np.meshgrid(z_grid, sigmas)
M = regularity_margin(Z, S)
im = ax.contourf(z_grid, sigmas, M, levels=50, cmap="RdYlGn")
ax.contour(z_grid, sigmas, M, levels=[0], colors="black", linewidths=1.5)
plt.colorbar(im, ax=ax, label="2 - R(z)·(σ+z)")
ax.set_xlabel("z = (log v − μ) / σ")
ax.set_ylabel("σ")
ax.set_title("Regularity margin  2 − R(z)·(σ+z)\nBlack line = boundary (=0)")

# Plot 3: for fixed sigma, show how mu shifts the effective z_floor
ax = axes[2]
v_grid = np.linspace(floor_price, 5, 500)
for mu in mu_values:
    dist = lognorm(s=sigma_check, scale=np.exp(mu))
    psi  = v_grid - (1 - dist.cdf(v_grid)) / dist.pdf(v_grid)
    ax.plot(v_grid, psi, lw=2, label=f"μ={mu}")
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(floor_price, color="red", linestyle=":", lw=1.5, label=f"floor={floor_price}")
ax.set_xlabel("v")
ax.set_ylabel("ψ(v)")
ax.set_title(f"ψ(v) for σ={sigma_check}, varying μ\n(v ≥ floor only)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# ── New plot: y = v  and  y = (1 - F(v)) / f(v) ──────────────────────────────
v_plot = np.linspace(0.01, 5, 500)
mu_vals_new   = [0.0, 0.5, 1.0]
sigma_vals_new = [0.5, 1.0, 1.5]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

# Subplot 1: y = v  (straight line, independent of mu/sigma)
ax = axes2[0]
ax.plot(v_plot, v_plot, color="steelblue", lw=2)
ax.set_xlabel("v")
ax.set_ylabel("y")
ax.set_title("y1 = v")

# Subplot 2: y = (1 - F(v)) / f(v) for various (mu, sigma)
ax = axes2[1]
for mu in mu_vals_new:
    for sigma in sigma_vals_new:
        dist = lognorm(s=sigma, scale=np.exp(mu))
        y = (1 - dist.cdf(v_plot)) / dist.pdf(v_plot)
        ax.plot(v_plot, y, lw=1.5, label=f"μ={mu}, σ={sigma}")
ax.set_xlabel("v")
ax.set_ylabel("y")
ax.set_title("y2 = (1 − F(v)) / f(v)")
ax.set_ylim(0, 10)
ax.legend(fontsize=11, ncol=2)

plt.tight_layout()
plt.show()

print(f"Critical sigma (global regularity): σ* ≈ {critical_sigma:.4f}" if critical_sigma else "No valid sigma found")
print(f"\nFor sigma={sigma_check}, monotone above floor={floor_price}?")
for mu in mu_values:
    z_floor = (np.log(floor_price) - mu) / sigma_check
    margin  = regularity_margin(z_grid[z_grid >= z_floor], sigma_check)
    print(f"  mu={mu}: min margin = {margin.min():.4f} → {'YES' if margin.min() > 0 else 'NO'}")
