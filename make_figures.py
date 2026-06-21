#!/usr/bin/env python3
"""
Regenerate paper figures and concentration indices from verified source tables.

Every value here is transcribed from a domain-winner table in the archive and
is traceable to a single source file (see SOURCES.md):

  35B prefill / generation winners:
    sae-tests/qwen-huahua-expert-identification/results/
      results_domain_specialists_20260408T235839Z.md
    (prefill "Domain winners" table; "Generation Trimmed" -> "Domain winners")

  122B prefill winners:
    moe-routing-organized/qwen3.5-122b-a10b-huahua/
      qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/results.md
    (prefill "Domain Winners")
  122B generation winners:
    .../qwen3.5-122B-A10B-huahua-domain-specialist-generation/results.md
    ("Generation Trimmed" -> "Domain winners")

  3-chunk token-balanced control (prefill<->generation top-W / top-S Jaccard):
    git-updates-moe/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/DOCS/RESULTS.md

The only computation is the deterministic concentration summary of each winner
list (distinct-winner count, max domains won by one expert, Herfindahl index,
and normalized Shannon entropy). No model is re-run here.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

DOMAINS = [
    "archaeology", "biology", "chemistry", "comparative_religion",
    "computer_science", "cybersecurity", "economics", "environmental_science",
    "history", "law", "linguistics", "mathematics", "medicine", "neuroscience",
    "philosophy", "physics", "political_science", "psychology",
    "software_engineering", "statistics",
]

# --- Verified per-domain winning expert (by routed weight W), one entry per ---
# --- domain, in the DOMAINS order above. -------------------------------------

# 35B Qwen3.5-35B-A3B (HauhauCS Q8_0), run 20260408T235839Z
W35_PREFILL = [224, 224, 130, 224, 224, 224, 224, 224, 224, 224,
               103, 224, 224, 224, 224, 224, 224, 224, 224, 224]
W35_GEN = [191, 250, 130, 170, 206, 188, 223, 8, 158, 48,
           103, 100, 152, 54, 114, 139, 224, 146, 61, 202]  # Generation Trimmed

# 122B Qwen3.5-122B-A10B (HauhauCS Q8_K_P)
W122_PREFILL = [233, 233, 233, 91, 233, 233, 215, 233, 9, 233,
                9, 233, 233, 233, 72, 233, 45, 131, 233, 233]
W122_GEN = [12, 181, 11, 159, 76, 174, 231, 159, 162, 226,
            102, 0, 42, 152, 40, 40, 160, 223, 139, 4]  # Generation Trimmed

# --- 3-chunk token-balanced control (446 prompt tokens / chunk) --------------
# prefill <-> generation top-W and top-S Jaccard overlap, per packed chunk
CHUNKS = ["A", "B", "C"]
JACCARD_W = [0.000000, 0.230769, 0.066667]
JACCARD_S = [0.066667, 0.230769, 0.000000]


def concentration(winners):
    n = len(winners)
    counts = {}
    for e in winners:
        counts[e] = counts.get(e, 0) + 1
    distinct = len(counts)
    max_wins = max(counts.values())
    herf = sum((c / n) ** 2 for c in counts.values())
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
    ent_norm = ent / math.log2(n)
    return dict(distinct=distinct, max_wins=max_wins, herfindahl=herf,
                entropy_bits=ent, entropy_norm=ent_norm)


SETS = [
    ("Qwen 35B", "prefill", W35_PREFILL),
    ("Qwen 35B", "generation", W35_GEN),
    ("Qwen 122B", "prefill", W122_PREFILL),
    ("Qwen 122B", "generation", W122_GEN),
]

STATS = {(m, p): concentration(w) for m, p, w in SETS}

C_PRE = "#1f4e79"   # prefill
C_GEN = "#c55a11"   # generation


# ---------------------------------------------------------------------------
# Figure 1: distinct domain winners and max domains won by one expert
# ---------------------------------------------------------------------------
def fig1():
    models = ["Qwen 35B", "Qwen 122B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    panels = [
        ("distinct", "distinct domain winners (of 20)", 21),
        ("max_wins", "domains won by the single top expert", 21),
    ]
    x = range(len(models))
    w = 0.38
    for ax, (key, ylab, ymax) in zip(axes, panels):
        pre = [STATS[(m, "prefill")][key] for m in models]
        gen = [STATS[(m, "generation")][key] for m in models]
        ax.bar([i - w / 2 for i in x], pre, w, color=C_PRE, label="prefill")
        ax.bar([i + w / 2 for i in x], gen, w, color=C_GEN, label="generation")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ymax)
        for i, v in enumerate(pre):
            ax.annotate(str(v), (i - w / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5)
        for i, v in enumerate(gen):
            ax.annotate(str(v), (i + w / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_distinct_winners.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_distinct_winners.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: concentration of the winner distribution (normalized entropy)
# ---------------------------------------------------------------------------
def fig2():
    models = ["Qwen 35B", "Qwen 122B"]
    pre = [STATS[(m, "prefill")]["entropy_norm"] for m in models]
    gen = [STATS[(m, "generation")]["entropy_norm"] for m in models]
    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.bar([i - w / 2 for i in x], pre, w, color=C_PRE, label="prefill")
    ax.bar([i + w / 2 for i in x], gen, w, color=C_GEN, label="generation")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("winner-distribution entropy (normalized, 0-1)")
    ax.set_title("Spread of domain-winning experts, prefill vs generation")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    for i, v in enumerate(pre):
        ax.annotate("%.2f" % v, (i - w / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5)
    for i, v in enumerate(gen):
        ax.annotate("%.2f" % v, (i + w / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_concentration.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_concentration.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: token-balanced control -- prefill<->generation expert overlap
# ---------------------------------------------------------------------------
def fig3():
    x = range(len(CHUNKS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.bar([i - w / 2 for i in x], JACCARD_W, w, color=C_PRE,
           label="top-W experts")
    ax.bar([i + w / 2 for i in x], JACCARD_S, w, color=C_GEN,
           label="top-S experts")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["chunk %s" % c for c in CHUNKS])
    ax.set_ylabel("prefill vs generation Jaccard overlap")
    ax.set_title("Token-balanced prompts (446 tokens each): "
                 "prefill and\ngeneration select different experts")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    for i, v in enumerate(JACCARD_W):
        ax.annotate("%.2f" % v, (i - w / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5)
    for i, v in enumerate(JACCARD_S):
        ax.annotate("%.2f" % v, (i + w / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_token_balanced.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_token_balanced.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote figures to", OUT)
    print()
    print("Concentration of domain-winner distribution (20 domains):")
    print("%-11s %-11s %8s %8s %10s %9s" %
          ("model", "phase", "distinct", "maxwins", "herfindahl", "ent_norm"))
    for m, p, _ in SETS:
        s = STATS[(m, p)]
        print("%-11s %-11s %8d %8d %10.3f %9.3f" %
              (m, p, s["distinct"], s["max_wins"], s["herfindahl"],
               s["entropy_norm"]))
