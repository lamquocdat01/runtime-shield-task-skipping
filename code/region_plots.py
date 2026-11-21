"""region_plots.py -- P1 Viec 1 (item 6): 2-D views of the extended certificate region from
results/region_ext/bruteforce_grid.parquet (exhaustive engine; IMITATOR agrees on its box).
Each panel shows R*(params) = the smallest certified R, or an excess over the naive additive
formula min(T, W'+1) + d + b. Fixed comparison '>=' unless stated.
Outputs: results/region_ext/{b_TW,d_block_TW,d_b_interaction,kappa_T,CB_hole,ship_shift}.png
Usage: python code/region_plots.py --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "region_ext"


def pivot(df, x, y, val):
    p = df.pivot_table(index=y, columns=x, values=val, aggfunc="first")
    return p.sort_index(ascending=True)


def heat(ax, p, title, cmap="viridis", fmt=None, vmin=None, vmax=None):
    im = ax.imshow(p.to_numpy(float), origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[p.columns.min() - .5, p.columns.max() + .5,
                           p.index.min() - .5, p.index.max() + .5])
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused; recorded")
    ap.parse_args()
    g = pd.read_parquet(OUT / "bruteforce_grid.parquet")
    one = g.kappa_str == "1"
    base = g[one & (g.cb == 0) & (g.d == 0) & (g["ge"] == 1)]

    # 1. frame drops: R* over (T, W) for b = 0, 1, 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, b in zip(axs, (0, 1, 2)):
        heat(ax, pivot(base[base.b == b], "T", "W", "R_star"),
             f"R*(T, W), b = {b} drops per R-window", vmin=1, vmax=62)
        ax.set_xlabel("T (refresh)")
        ax.set_ylabel("W (reuse cap)")
        TT = np.arange(1, 31)
        ax.plot(TT, (b + 1) * TT, "w--", lw=0.8)    # regime boundary (b+1)T = W
        ax.set_ylim(-.5, 20.5)
    fig.suptitle("Theorem 2'(b): R* = (b+1)T if (b+1)T <= W, else W+1+b-floor(W/T)  "
                 "(dashed: (b+1)T = W)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "b_TW.png", dpi=150)
    plt.close(fig)

    # 2. blocking detector: excess of block over pipelined, d = 3
    sub = g[one & (g.cb == 0) & (g.b == 0) & (g["ge"] == 1) & (g.d == 3)]
    pb = pivot(sub[sub.block == 1], "T", "W", "R_star")
    pp = pivot(sub[sub.block == 0], "T", "W", "R_star")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    heat(ax, pb - pp, "R*_block - R*_pipe, d = 3 (refresh shadowing / cap shadowing)",
         cmap="magma")
    ax.set_xlabel("T")
    ax.set_ylabel("W")
    fig.tight_layout()
    fig.savefig(OUT / "d_block_TW.png", dpi=150)
    plt.close(fig)

    # 3. d x b interaction (blocking), two (T, W) points
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (T, W) in zip(axs, ((15, 4), (4, 12))):
        s = g[one & (g.cb == 0) & (g["ge"] == 1) & (g["T"] == T) & (g.W == W)
              & ((g.block == 1) | (g.d <= 1))]
        p = pivot(s, "b", "d", "R_star")
        inter = p.sub(p.iloc[:, 0], axis=0).sub(p.iloc[0, :], axis=1) + p.iloc[0, 0]
        heat(ax, inter, f"interaction R*(d,b)-R*(d,0)-R*(0,b)+R*(0,0), T={T}, W={W}",
             cmap="coolwarm", vmin=-6, vmax=6)
        for (i, dd), (j, bb) in [((i, dd), (j, bb)) for i, dd in enumerate(p.index)
                                 for j, bb in enumerate(p.columns)]:
            ax.text(bb, dd, int(p.loc[dd, bb]), ha="center", va="center", fontsize=7)
        ax.set_xlabel("b (drops)")
        ax.set_ylabel("d (blocking delay)")
    fig.suptitle("numbers: R*; colour: non-additive part (0 = additive)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "d_b_interaction.png", dpi=150)
    plt.close(fig)

    # 4. kappa: R* over (kappa, T) at W = 20 (refresh-limited regime)
    s = g[(g.cb == 0) & (g.d == 0) & (g.b == 0) & (g["ge"] == 1) & (g.W == 20)]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for kap, x in s.groupby("kappa"):
        x = x.sort_values("T")
        ax.step(x["T"], x.R_star, where="mid", lw=1, label=f"kappa = {x.kappa_str.iloc[0]}")
    ax.set_xlabel("T (refresh-clock units)")
    ax.set_ylabel("R*  (= ceil(T/kappa) here)")
    ax.set_title("Theorem 2'(kappa), W = 20, b = 0", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "kappa_T.png", dpi=150)
    plt.close(fig)

    # 5. circuit-breaker hole: R*_CB - R*_noCB (blocking, d = 2..5), as deployed point marked
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, d in zip(axs, (2, 4)):
        s = g[one & (g.b == 0) & (g["ge"] == 1) & (g.d == d) & (g.block == 1)]
        p = pivot(s[s.cb == 1], "T", "W", "R_star") - pivot(s[s.cb == 0], "T", "W", "R_star")
        heat(ax, p, f"R*(G1+CB) - R*(G1), blocking d = {d}", cmap="Reds", vmin=0, vmax=4)
        ax.plot([15], [4], "c*", ms=10)
        ax.set_xlabel("T")
        ax.set_ylabel("W")
    fig.suptitle("composition with the ASMAG circuit breaker shrinks the region "
                 "(frozen fsl + busy detector); * = deployed (15, 4)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "CB_hole.png", dpi=150)
    plt.close(fig)

    # 6. the shipped bug as a unit shift of the region (b = 0, d = 0)
    s = g[one & (g.cb == 0) & (g.d == 0) & (g.b == 0)]
    p = pivot(s[s["ge"] == 0], "T", "W", "R_star") - pivot(s[s["ge"] == 1], "T", "W", "R_star")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    heat(ax, p, "R*('>') - R*('>='): 1 where W+1 < T, else 0", cmap="Greys", vmin=0, vmax=1)
    ax.plot([15], [4], "r*", ms=10)
    ax.set_xlabel("T")
    ax.set_ylabel("W")
    fig.tight_layout()
    fig.savefig(OUT / "ship_shift.png", dpi=150)
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
