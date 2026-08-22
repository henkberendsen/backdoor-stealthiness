#!/usr/bin/env python3
"""
Analysis #6(a) - Cross-architecture stability of FEATURE-space rankings (S&P, appendix).

Question: do the tri-space conclusions hold across ResNet18 / VGG16 / ViT-Small?
Cross-arch data we actually have (no retraining): input metrics are architecture-INDEPENDENT
(identical across archs, tau=1 by construction); feature SS/DSWD exist for all three archs on
CIFAR-10 (8 attacks; DFST not trained on VGG/ViT, DFBA has no SS); parameter metrics are
CNN-only and VGG has only 3 attacks -> reported as a limitation, not computed here.

We compute Kendall's tau between the per-architecture attack rankings for SS, DSWD, and a
combined feature footprint (mean of SS- and DSWD-rank), over the 8 shared attacks on CIFAR-10.

Sources (with provenance, cross-checked):
  ResNet18 : analysis/metrics_matrix.csv  (CIFAR-10, high PR)
  VGG/ViT  : feature_steathiness_{vgg,vit}_cifar10.txt  for the 6 non-clean-label attacks;
             VGG Narcissus/Grond from paper Table 14 (ViT has them in the .txt).
Outputs (analysis/analysis6/): arch_stability_feature.csv (provenance),
  arch_feature_scatter.pdf (figure). Paper-facing table+prose: cross_arch.tex (hand-maintained).
"""
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ATTACKS = ["badnet", "blended", "wanet", "bpp", "adaptive_patch", "adaptive_blend",
           "narcissus", "grond"]                       # 8 shared attacks (no DFST/DFBA)
PRETTY = {"badnet": "BadNets", "blended": "Blend", "wanet": "WaNet", "bpp": "BppAttack",
          "adaptive_patch": "Adap-P", "adaptive_blend": "Adap-B",
          "narcissus": "Narcissus", "grond": "Grond"}


def parse_txt(path):
    """feature_steathiness_*_cifar10.txt -> {'SS': {atk:val}, 'DSWD': {atk:val}} keyed by bare attack."""
    out = {"SS": {}, "DSWD": {}}
    for line in Path(path).read_text().splitlines():
        if line.startswith("feature_results_ss:"):
            d = ast.literal_eval(line.split(":", 1)[1].strip())
            out["SS"] = {k.replace("_p0.05", ""): v for k, v in d.items()}
        elif line.startswith("feature_results_dswd:"):
            d = ast.literal_eval(line.split(":", 1)[1].strip())
            out["DSWD"] = {k.replace("_p0.05", ""): v for k, v in d.items()}
    return out


# ResNet18 from the verified matrix (CIFAR-10, high PR)
m = pd.read_csv(REPO / "analysis" / "metrics_matrix.csv")
r = m[(m.Dataset == "CIFAR-10") & (m.PR_level == "high")].set_index("Attack")
RES = {"SS": {a: float(r.loc[a, "SS"]) for a in ATTACKS},
       "DSWD": {a: float(r.loc[a, "DSWD"]) for a in ATTACKS}}

vgg_txt = parse_txt(REPO / "feature_steathiness_vgg_cifar10.txt")
vit_txt = parse_txt(REPO / "feature_steathiness_vit_cifar10.txt")

# VGG: 6 from .txt; Narcissus/Grond from paper Table 14
VGG = {"SS": dict(vgg_txt["SS"]), "DSWD": dict(vgg_txt["DSWD"])}
VGG["SS"].update({"narcissus": 0.459, "grond": 0.317})
VGG["DSWD"].update({"narcissus": 3.360, "grond": 2.340})
VIT = {"SS": vit_txt["SS"], "DSWD": vit_txt["DSWD"]}          # all 8 in the .txt

ARCH = {"ResNet": RES, "VGG": VGG, "ViT": VIT}

# sanity: every arch must have all 8 attacks for SS and DSWD
for a, d in ARCH.items():
    for mt in ["SS", "DSWD"]:
        miss = [k for k in ATTACKS if k not in d[mt]]
        assert not miss, f"{a} {mt} missing {miss}"
print("All three architectures cover the 8 shared attacks for SS and DSWD.")

# build per-arch vectors in fixed attack order
def vec(arch, mt):
    return np.array([ARCH[arch][mt][a] for a in ATTACKS], float)

def rank(v):
    return pd.Series(v).rank().values

# combined feature footprint = mean of SS-rank and DSWD-rank
COMB = {a: (rank(vec(a, "SS")) + rank(vec(a, "DSWD"))) / 2 for a in ARCH}

pairs = [("ResNet", "VGG"), ("ResNet", "ViT"), ("VGG", "ViT")]
rows = []
for name, getter in [("Input (by construction)", None),
                     ("Feature: SS", lambda a: vec(a, "SS")),
                     ("Feature: DSWD", lambda a: vec(a, "DSWD")),
                     ("Feature: combined", lambda a: COMB[a]),
                     ("Parameter (CNN-only)", None)]:
    row = {"Space/metric": name}
    for x, y in pairs:
        if name.startswith("Input"):
            row[f"{x}-{y}"] = 1.00
        elif name.startswith("Parameter"):
            row[f"{x}-{y}"] = np.nan          # VGG=3 attacks only; ViT N/A
        else:
            row[f"{x}-{y}"] = round(kendalltau(getter(x), getter(y))[0], 2)
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(HERE / "arch_stability_feature.csv", index=False)
print(T.to_string(index=False))
print("\nSpearman (combined feature):",
      {f"{x}-{y}": round(spearmanr(COMB[x], COMB[y])[0], 2) for x, y in pairs})

# NOTE: the paper-facing table (tab:arch_stability) is hand-maintained inside cross_arch.tex;
# arch_stability_feature.csv above is its provenance. No standalone table .tex is emitted.

# ---- figure: ResNet feature rank vs VGG / ViT (combined) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    rr = COMB["ResNet"]
    for other, mk in [("VGG", "o"), ("ViT", "s")]:
        ax.scatter(rr, COMB[other], marker=mk, s=45, label=f"{other} ($\\tau$={kendalltau(rr, COMB[other])[0]:+.2f})")
        for i, a in enumerate(ATTACKS):
            ax.annotate(PRETTY[a], (rr[i], COMB[other][i]), fontsize=6, alpha=0.7)
    lim = [0.5, 8.5]
    ax.plot(lim, lim, ls=":", color="gray", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ResNet18 feature rank (1 = stealthiest)")
    ax.set_ylabel("VGG16 / ViT-Small feature rank")
    ax.set_title("Cross-architecture feature ranking (CIFAR-10)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "arch_feature_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {HERE/'arch_feature_scatter.pdf'}")
except Exception as e:
    print("[figure skipped]", e)
