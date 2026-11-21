"""paper_figs.py -- the figures of the manuscript, from committed result files only.
  fig_region.pdf    R*(T, W) of Theorem 2' for three factor settings (fixed '>=')
  fig_cb.pdf        Prop. 2.7: R*(G1 + breaker) - R*(G1) over the probe constants (K, P),
                    deployed point (T, W) = (15, 4), blocking detector d = 2, 3, 4
  fig_frontier.pdf  energy-miss plane: correct configurations, Pareto frontier, G1-family
                    mutants (kept after the equivalence filter), MC-violating ones highlighted
Colour: one sequential blue ramp for magnitude; categorical slots 1-2 (blue, orange) for
identity, always with a second encoding (marker shape) and a legend.
Usage: python code/paper_figs.py --seed 42 [--out paper/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

TOPIC_DIR = Path(__file__).resolve().parent.parent
RES = TOPIC_DIR / "results"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95",
         "#104281", "#0d366b"]
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def seq_cmap(vmax: int):
    """Discrete sequential map with one colour per integer level 0..vmax (or bands)."""
    n = len(BLUES)
    bounds = np.linspace(-0.5, vmax + 0.5, min(vmax + 1, n) + 1)
    cols = [BLUES[int(round(i * (n - 1) / max(1, len(bounds) - 2)))] for i in range(len(bounds) - 1)]
    return ListedColormap(cols), BoundaryNorm(bounds, len(cols))


def fig_region(out: Path) -> None:
    g = pd.read_parquet(RES / "region_ext" / "bruteforce_grid.parquet")
    g = g[(g["ge"] == 1) & (g.kappa_str == "1") & (g.cb == 0)]
    panels = [("(a) $d=0$, $b=0$", (g.d == 0) & (g.b == 0) & (g.block == 0)),
              ("(b) $d=0$, $b=2$ drops per window", (g.d == 0) & (g.b == 2) & (g.block == 0)),
              ("(c) blocking detector $d=3$, $b=0$", (g.d == 3) & (g.b == 0) & (g.block == 1))]
    vmax = 0
    mats = []
    for _, m in panels:
        s = g[m]
        M = s.pivot(index="W", columns="T", values="R_star").sort_index()
        mats.append(M)
        vmax = max(vmax, int(M.values.max()))
    bounds = [b for b in (1, 3, 5, 7, 9, 12, 16, 21, 27) if b <= vmax] + [vmax + 1]
    cols = [BLUES[round(i * (len(BLUES) - 1) / max(1, len(bounds) - 2))]
            for i in range(len(bounds) - 1)]
    cmap, norm = ListedColormap(cols), BoundaryNorm(bounds, len(cols))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True, constrained_layout=True)
    for ax, (title, _), M in zip(axs, panels, mats):
        im = ax.pcolormesh(M.columns, M.index, M.values, cmap=cmap, norm=norm, shading="nearest",
                           edgecolors="face", linewidth=0.3, rasterized=True)
        ax.plot([15], [4], marker="*", ms=9, mfc="white", mec=INK, mew=0.8, ls="none")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("refresh period $T$")
        ax.tick_params(length=2)
    T = np.arange(1, 31)
    axs[1].plot(T, 3 * T, ls="--", lw=1, color=INK2)
    axs[1].set_ylim(-0.5, 20.5)
    axs[1].text(7.5, 17.5, "$(b{+}1)T = W$", color=INK, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    axs[0].set_ylabel("reuse cap $W$")
    cb = fig.colorbar(im, ax=axs, shrink=0.95, pad=0.01)
    cb.set_label("minimal certified bound $R^*$")
    fig.savefig(out / "fig_region.pdf", metadata={"CreationDate": None})  # byte-stable (C8b)
    plt.close(fig)


def fig_cb(out: Path) -> None:
    df = pd.read_parquet(RES / "region_ext" / "cb_rule_ext.parquet")
    df = df[(df["T"] == 15) & (df.W == 4) & (df["ge"] == 1) & (df.allphase == 1) & (df.K <= 12)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.75), sharey=True, constrained_layout=True)
    cmap = ListedColormap(["#f0efec", BLUES[2], BLUES[5], BLUES[8]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    for ax, d in zip(axs, (2, 3, 4)):
        s = df[df.d == d]
        M = (s.G_CB - s.G_G1).to_numpy()
        H = np.full((12, 12), np.nan)
        for (K, P), v in zip(zip(s.K, s.P), M):
            H[P - 1, K - 1] = v
        im = ax.pcolormesh(np.arange(1, 13), np.arange(1, 13), H, cmap=cmap, norm=norm,
                           shading="nearest", edgecolors="white", linewidth=0.6)
        for K in range(2, 13):
            for P in range(1, K):
                if np.gcd(K, P) % d == 0:
                    ax.plot(K, P, "o", ms=3.2, color=INK, mec="white", mew=0.4)
        ax.set_title(f"blocking detector $d={d}$", loc="left", color=INK)
        ax.set_xlabel("probe period $K$")
        ax.set_xlim(1.5, 12.5)
        ax.set_ylim(0.5, 11.5)
        ax.set_aspect("equal")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([1, 3, 5, 7, 9, 11])
        ax.tick_params(length=2)
    axs[0].set_ylabel("probe length $P$")
    cb = fig.colorbar(im, ax=axs, ticks=[0, 1, 2, 3], shrink=0.9, pad=0.01)
    cb.set_label("$R^*$ increase from the breaker")
    fig.savefig(out / "fig_cb.pdf", metadata={"CreationDate": None})  # byte-stable (C8b)
    plt.close(fig)


def fig_frontier(out: Path) -> None:
    f = pd.read_parquet(RES / "frontier_correct.parquet")
    m = pd.read_csv(RES / "mutants_v11.csv")
    m = m[(m.family == "G1") & m.kept_union]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.scatter(f.E, 100 * f.miss_all, s=4, color="#9e9d98", lw=0, label="correct configurations", zorder=1)
    p = f[f.is_pareto].sort_values("E")
    ax.plot(p.E, 100 * p.miss_all, lw=2, color=C1, label="Pareto frontier of correct configs",
            zorder=2)
    ok = m[m.mc_violates != True]  # noqa: E712
    bad = m[m.mc_violates == True]  # noqa: E712
    ax.scatter(ok.E, 100 * ok.miss_all, s=22, marker="o", facecolor="white", edgecolor=INK2,
               lw=0.9, label="mutant, $I_R$ holds (MC)", zorder=3)
    ax.scatter(bad.E, 100 * bad.miss_all, s=34, marker="^", color=C2, edgecolor="white", lw=0.8,
               label="mutant, violates $I_R$ (MC)", zorder=4)
    info = __import__("json").loads((RES / "frontier_correct_info.json").read_text())
    dp = info["deployed_points"]
    ax.scatter([dp["fixed_ge"]["E"]], [100 * dp["fixed_ge"]["miss_all"]], s=40, marker="s",
               color=C1, edgecolor="white", lw=0.8, zorder=5, label="fixed controller (15, 4)")
    ship = m[m.mutant == "M1a"].iloc[0]
    ax.annotate("shipped fault", (ship.E, 100 * ship.miss_all), xytext=(ship.E - 9, 100 * ship.miss_all + 1.6),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    lo, hi = 222, 248
    ax.set_xlim(lo, hi)
    sel = f[(f.E > lo) & (f.E < hi)]
    ax.set_ylim(100 * sel.miss_all.min() - 0.6, 100 * max(sel.miss_all.max(), bad.miss_all.max()) + 0.8)
    ax.set_xlabel("energy $E$ (mJ/frame, linear energy model)")
    ax.set_ylabel("event miss rate (%)")
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.savefig(out / "fig_frontier.pdf", metadata={"CreationDate": None})  # byte-stable (C8b)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (deterministic); recorded")
    ap.add_argument("--out", default=str(TOPIC_DIR / "paper" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_region(out)
    fig_cb(out)
    fig_frontier(out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
