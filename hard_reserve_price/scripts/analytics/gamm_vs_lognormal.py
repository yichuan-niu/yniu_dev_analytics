import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 10, 1000)

# Gamma: tuned so each peaks just left of its lognormal pair
# peaks at x=(shape-1)*scale: ~0.5, ~0.2, ~2.0 vs lognormal peaks ~0.78, ~0.37, ~2.12
gamma_params = [(2, 0.5), (1.5, 0.4), (5, 0.5)]
# Lognormal: mu, sigma (of underlying normal)
lognorm_params = [(0, 0.5), (0, 1.0), (1, 0.5)]

fig, ax = plt.subplots(figsize=(10, 5))

colors = ["tab:blue", "tab:orange", "tab:green"]

for (mu, sigma), color in zip(lognorm_params, colors):
    rv = stats.lognorm(s=sigma, scale=np.exp(mu))
    ax.plot(x, rv.pdf(x), linestyle="-", color=color, label=f"Lognormal μ={mu}, σ={sigma}")

for (shape, scale), color in zip(gamma_params, colors):
    rv = stats.gamma(a=shape, scale=scale)
    ax.plot(x, rv.pdf(x), linestyle=":", linewidth=2, color=color, label=f"Gamma shape={shape}, scale={scale}")

ax.set_xlabel("x")
ax.set_ylabel("PDF")
ax.set_xlim(0, 10)
ax.set_ylim(0)
ax.legend()

fig.suptitle("Gamma vs Lognormal Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()