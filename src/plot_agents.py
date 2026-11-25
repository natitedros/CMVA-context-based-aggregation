"""Presentation figures for the self-distilled agents.

    python src/plot_agents.py

Reads results/ on Drive and writes three PNGs to results/figures/:
  1_own_value_gain.png    each agent's gain over the base model on its own value, with 95% intervals
  2_value_specificity.png gain of every agent on every value (both arms), own value outlined
  3_lora_cost_vs_gain.png LoRA as a share of full fine-tuning: cost and benefit on one scale
"""
import json
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from config import RESULTS_DIR, SEED, VALUES

FIG_DIR = f"{RESULTS_DIR}/figures"
FULL, LORA, CONTROL = "#257de0", "#f47230", "#9c9c9c"   # validated pair + a neutral reference
INK, INK_2, GRID = "#191919", "#666666", "#e6e6e6"
BLUE_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 14,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})

gens = lambda tag: pd.read_csv(f"{RESULTS_DIR}/generations_{tag}.csv")
align = lambda tag: json.load(open(f"{RESULTS_DIR}/alignment_{tag}.json"))
agent = lambda tag: json.load(open(f"{RESULTS_DIR}/agent_{tag}.json"))


def gain_ci(tag, value, n_boot=2000):
    """Mean per-question gain over the base model on one value, with a 95% bootstrap interval."""
    rng = np.random.default_rng(SEED)
    d = (gens(tag)[f"p_{value}"] - gens("pt")[f"p_{value}"]).to_numpy()
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def fig_own_value_gain():
    series = [("Full fine-tune", "full_{}_sd", FULL), ("LoRA", "lora_{}_sd", LORA),
              ("Control (random answers)", "lora_control_sd", CONTROL)]
    x, w = np.arange(len(VALUES)) * 1.15, 0.3
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, (label, pattern, color) in enumerate(series):
        stats = [gain_ci(pattern.format(v), v) for v in VALUES]
        mean = np.array([s[0] for s in stats])
        err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
        pos = x + (i - 1) * w
        ax.bar(pos, mean, w - 0.03, color=color, label=label, zorder=2)
        ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.5, capsize=4, zorder=3)
        if label != series[2][0]:  # direct labels on the two arms; the control stays quiet
            for p, m, hi in zip(pos, mean, err[1]):
                ax.text(p, m + hi + 0.002, f"+{m:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(x, [v.capitalize() for v in VALUES])
    ax.set_ylabel("Gain over base model (expected reward)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_title("Every agent beats the base model on its own value", loc="left", pad=36)
    ax.text(0, 1.03, "Mean gain on 500 held-out questions; bars show 95% bootstrap intervals",
            transform=ax.transAxes, color=INK_2, fontsize=13)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(-0.02, ax.get_ylim()[1] + 0.015)
    return fig


def fig_value_specificity():
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    base = align("pt")
    grids = {}
    for arm in ["full", "lora"]:
        grids[arm] = np.array([[align(f"{arm}_{a}_sd")[v]["expected_reward"] - base[v]["expected_reward"]
                                for a in VALUES] for v in VALUES])
    vmax = max(g.max() for g in grids.values())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"wspace": 0.35})
    for ax, (arm, title) in zip(axes, [("full", "Full fine-tune"), ("lora", "LoRA")]):
        g = grids[arm]
        im = ax.imshow(g, cmap=cmap, vmin=0, vmax=vmax)
        for r in range(len(VALUES)):
            for c in range(len(VALUES)):
                ax.text(c, r, f"+{g[r, c]:.3f}", ha="center", va="center", fontsize=12,
                        color="white" if g[r, c] > 0.55 * vmax else INK)
            ax.add_patch(Rectangle((r - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2.5))
        ax.set_xticks(range(len(VALUES)), [v.capitalize() for v in VALUES])
        ax.set_yticks(range(len(VALUES)), [v.capitalize() for v in VALUES])
        ax.set_xlabel("Agent trained for")
        if arm == "full":
            ax.set_ylabel("Scored on")
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Gain over base model")
    fig.suptitle("Each agent improves its own value the most (outlined)", x=0.06, ha="left", fontsize=18)
    return fig


def fig_cost_vs_gain():
    full = [agent(f"full_{v}_sd") for v in VALUES]
    lora = [agent(f"lora_{v}_sd") for v in VALUES]
    base_mb = full[0]["saved_size_mb"]  # a full agent in fp16 is the size of one base model
    full_gain = np.mean([gain_ci(f"full_{v}_sd", v)[0] for v in VALUES])
    lora_gain = np.mean([gain_ci(f"lora_{v}_sd", v)[0] for v in VALUES])
    rows = [
        ("Trainable parameters", lora[0]["trainable_params"] / full[0]["trainable_params"]),
        ("Peak training memory", np.mean([l["peak_train_gpu_mb"] for l in lora]) / np.mean([f["peak_train_gpu_mb"] for f in full])),
        ("Storage for all five agents", (base_mb + sum(l["saved_size_mb"] for l in lora)) / sum(f["saved_size_mb"] for f in full)),
        ("Training time", np.mean([l["train_seconds"] for l in lora]) / np.mean([f["train_seconds"] for f in full])),
        ("Own-value gain", lora_gain / full_gain),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    y = np.arange(len(rows))[::-1]
    for yi, (label, share) in zip(y, rows):
        ax.barh(yi, share * 100, 0.6, color=FULL if label != "Own-value gain" else LORA, zorder=2)
        text = f"{share:.0%}" if share >= 0.1 else f"{share:.1%}"
        if share > 0.9:  # long bars: label inside the end, so it doesn't cross the 100% line
            ax.text(share * 100 - 1.5, yi, text, va="center", ha="right", fontsize=14, color="white")
        else:
            ax.text(share * 100 + 1.5, yi, text, va="center", fontsize=14, color=INK)
    ax.axvline(100, color=INK_2, linestyle="--", linewidth=1.5, zorder=1)
    ax.text(100, y[0] + 0.55, "Full fine-tune = 100%", ha="center", fontsize=12, color=INK_2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 118)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.set_xlabel("LoRA as a share of full fine-tuning")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"LoRA keeps {rows[-1][1]:.0%} of the gain at a fraction of the cost", loc="left", pad=16)
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, make in [("1_own_value_gain", fig_own_value_gain),
                       ("2_value_specificity", fig_value_specificity),
                       ("3_lora_cost_vs_gain", fig_cost_vs_gain)]:
        fig = make()
        fig.savefig(f"{FIG_DIR}/{name}.png")
        plt.close(fig)
        print(f"saved {FIG_DIR}/{name}.png")


if __name__ == "__main__":
    main()
