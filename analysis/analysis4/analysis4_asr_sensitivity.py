#!/usr/bin/env python3
"""
Analysis #4 - Low-ASR sensitivity (S&P resubmission).

Question: are our stealthiness conclusions artifacts of failed (low-ASR)
attacks? We recompute the main per-space attack rankings under three filters
  D_all (all configs), D_60 (ASR>=60%), D_80 (ASR>=80%)
and check whether (i) the best-per-space attacks, (ii) the "no attack is stealthy in all
three spaces" conclusion, and (iii) the CDBI/TUP top attacks are stable.

Method (consistent with Analysis #1, within-dataset to remove dataset-scale confounds):
  * orient all 16 metrics so higher = larger footprint (invert PSNR,SSIM,pHash,CDBI);
  * within each dataset, percentile-rank configs per metric (0 = stealthiest, 1 = loudest)
    over the configs that PASS the filter;
  * per (attack, space) score = mean percentile over that attack's surviving configs x the
    space's metrics (lower = stealthier). Best-in-space = argmin.
  * "uniform" stealth check: an attack's worst-space score = max over spaces; the most
    balanced attack minimises it. No attack is uniformly stealthy iff that minimum is high.
  * stability: Kendall tau between the attack ordering under D_all and under each filter.

Input : analysis/metrics_matrix.csv   Output (all in analysis/analysis4/):
  asr_sensitivity.csv (provenance), asr_filter_rank_stability.pdf (figure).
The paper-facing tables + prose live in the hand-maintained asr_sensitivity.tex.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX = HERE.parent / "metrics_matrix.csv"

METR = ["l1", "l2", "l_inf", "MSE", "PSNR", "SSIM", "LPIPS", "IS", "pHash", "SAM",
        "SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP"]
INVERT = {"PSNR", "SSIM", "pHash", "CDBI"}
SPACES = {"input": METR[:10], "feature": ["SS", "DSWD", "CDBI"], "param": ["UCLC", "TAC", "TUP"]}
FILTERS = [("All configs", 0), ("ASR$\\ge$60\\%", 60), ("ASR$\\ge$80\\%", 80)]
PRETTY = {"badnet": "BadNets", "blended": "Blend", "wanet": "WaNet", "bpp": "BppAttack",
          "adaptive_patch": "Adap-Patch", "adaptive_blend": "Adap-Blend", "dfst": "DFST",
          "narcissus": "Narcissus", "grond": "Grond", "dfba": "DFBA"}

try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None


def oriented(df):
    F = df.copy()
    for c in INVERT:
        F[c] = -F[c]
    return F


def space_scores(df):
    """Return DataFrame indexed by attack: per-space mean percentile + per-metric + support n."""
    d = df.copy()
    for c in METR:
        d[c + "_r"] = d.groupby("Dataset")[c].rank(pct=True)   # 0=stealthiest,1=loudest in dataset
    rows = {}
    for atk, g in d.groupby("Attack"):
        r = {sp: g[[c + "_r" for c in mets]].stack().mean() for sp, mets in SPACES.items()}
        r["CDBI"] = g["CDBI_r"].mean()
        r["TUP"] = g["TUP_r"].mean()
        r["overall"] = np.nanmean([r["input"], r["feature"], r["param"]])
        r["n"] = len(g)
        rows[atk] = r
    return pd.DataFrame(rows).T


def main():
    base = oriented(pd.read_csv(MATRIX))
    scores = {th: space_scores(base[base.ASR >= th]) for _, th in FILTERS}   # keyed by threshold

    # ---- main conclusions table ----
    def best(S, col):
        return PRETTY[S[col].astype(float).idxmin()]
    worst_space = lambda S: S[["input", "feature", "param"]].astype(float).max(axis=1)

    rows = [{"Conclusion": lab} for lab in
            ["Stealthiest input-space footprint", "Stealthiest feature-space footprint",
             "Stealthiest parameter-space footprint", "Same attack best in all 3 spaces?",
             "Most balanced attack (min worst space)", "CDBI top-ranked attack",
             "TUP top-ranked attack"]]
    table = pd.DataFrame(rows)
    for label, th in FILTERS:
        S = scores[th]
        wc = worst_space(S)
        all3 = "No" if len({S.input.astype(float).idxmin(), S.feature.astype(float).idxmin(),
                            S.param.astype(float).idxmin()}) > 1 else "Yes"
        table[label] = [best(S, "input"), best(S, "feature"), best(S, "param"), all3,
                        f"{PRETTY[wc.idxmin()]} ({wc.min():.2f})", best(S, "CDBI"), best(S, "TUP")]
    table.to_csv(HERE / "asr_sensitivity.csv", index=False)
    print(table.to_string(index=False))

    # ---- support (how many configs survive per attack) ----
    print("\nsurviving configs per attack:")
    print(pd.DataFrame({th: scores[th]["n"] for _, th in FILTERS}).astype(int).to_string())

    # ---- rank-stability (Kendall tau of attack ordering vs All) ----
    stab = {}
    if kendalltau:
        for sp in ["input", "feature", "param", "overall"]:
            a = scores[0][sp].astype(float)
            stab[sp] = {}
            for _, th in FILTERS[1:]:
                b = scores[th][sp].astype(float)
                idx = a.dropna().index.intersection(b.dropna().index)
                stab[sp][th] = kendalltau(a.loc[idx], b.loc[idx])[0]
        print("\nKendall tau vs All:")
        print(pd.DataFrame(stab).T.round(2).to_string())

    # NOTE: the paper-facing tables (tab:asr_sensitivity, tab:asr_stability) are hand-maintained
    # inside asr_sensitivity.tex. asr_sensitivity.csv + the printed Kendall-tau values above are
    # their provenance; we do not emit standalone table .tex files (avoids duplicate labels).

    # ---- figure: per-space rank-stability bump charts (one panel per space) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[skip figure: matplotlib unavailable]")
        return
    xs = list(range(len(FILTERS)))
    attacks = list(scores[0].index)
    cmap = plt.get_cmap("tab10")
    colour = {atk: cmap(i % 10) for i, atk in enumerate(attacks)}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), sharey=True)
    for ax, sp in zip(axes, ["input", "feature", "param"]):
        ranks = {th: scores[th][sp].astype(float).rank(method="min") for _, th in FILTERS}
        for atk in attacks:
            y = [ranks[th].get(atk, np.nan) for _, th in FILTERS]
            if all(np.isnan(v) for v in y):       # e.g. DFBA has no feature score
                continue
            ax.plot(xs, y, "-o", color=colour[atk], lw=1.6, ms=4)
            if not np.isnan(y[-1]):
                ax.text(xs[-1] + 0.05, y[-1], PRETTY[atk], va="center", fontsize=7, color=colour[atk])
        ax.set_title({"input": "Input space", "feature": "Feature space",
                      "param": "Parameter space"}[sp], fontsize=10)
        ax.set_xticks(xs); ax.set_xticklabels(["All", "≥60%", "≥80%"])
        ax.invert_yaxis(); ax.set_xlim(-0.15, xs[-1] + 1.0)
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylabel("Footprint rank  (1 = stealthiest)")
    axes[0].set_yticks(range(1, len(attacks) + 1))
    fig.suptitle("Attack rank stability under ASR filtering (per space)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "asr_filter_rank_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {HERE/'asr_filter_rank_stability.pdf'}")


if __name__ == "__main__":
    main()
