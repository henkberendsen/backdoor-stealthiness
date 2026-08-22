#!/usr/bin/env python3
"""
Analysis #1 - Metric disagreement matrix (S&P resubmission).

Thesis: stealthiness is a multi-view measurement problem, not a single scalar.
We orient every metric so HIGHER = larger footprint = LESS stealthy, then compute
pairwise Spearman rank correlations between the 16 footprint metrics over all
ResNet18 attack configurations.

Inputs : analysis/metrics_matrix.csv   (built by build_metrics_matrix.py)
Outputs (all in analysis/analysis1/): metric_rank_correlation.pdf (pooled heatmap = paper
         Figure), metric_rank_correlation_stratified.pdf (per-dataset-averaged),
         metric_corr_pooled.csv (16x16 rho), metric_agreement.csv (within/cross-space means).
The paper-facing section (figure + table + prose) is the hand-maintained metric_disagreement.tex;
metric_agreement.csv is the provenance for its numbers.

Robust to missing scipy/matplotlib: falls back to a numpy Spearman (no p-values)
and skips the figure, still writing all CSVs.
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent           # analysis/analysis1
ANALYSIS = HERE.parent                            # analysis/  (shared backbone lives here)
MATRIX = ANALYSIS / "metrics_matrix.csv"          # shared input
FIGDIR = HERE                                     # all Analysis-1 outputs stay inside analysis1/
ANADIR = HERE

METRICS = ["l1", "l2", "l_inf", "MSE", "PSNR", "SSIM", "LPIPS", "IS", "pHash", "SAM",
           "SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP"]
SPACE = {**{m: "input" for m in METRICS[:10]},
         **{m: "feature" for m in ["SS", "DSWD", "CDBI"]},
         **{m: "param" for m in ["UCLC", "TAC", "TUP"]}}
# higher value of these means MORE stealthy -> invert so higher = larger footprint
INVERT = {"PSNR", "SSIM", "pHash", "CDBI"}
LABEL = {"l1": "$\\ell_1$", "l2": "$\\ell_2$", "l_inf": "$\\ell_\\infty$", "MSE": "MSE",
         "PSNR": "PSNR", "SSIM": "SSIM", "LPIPS": "LPIPS", "IS": "IS", "pHash": "pHash",
         "SAM": "SAM", "SS": "SS", "DSWD": "DSWD", "CDBI": "CDBI", "UCLC": "UCLC",
         "TAC": "TAC", "TUP": "TUP"}

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def oriented(df):
    X = df[METRICS].astype(float).copy()
    for c in INVERT:
        X[c] = -X[c]
    return X


def corr_p(a, b):
    """Spearman rho and p over pairwise-complete observations."""
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 4:
        return np.nan, np.nan
    if HAVE_SCIPY:
        r, p = spearmanr(a[ok], b[ok])
        return float(r), float(p)
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(ra, rb)[0, 1])
    n = ok.sum()                                   # t-approx p-value
    if abs(r) >= 1:
        p = 0.0
    else:
        t = r * np.sqrt((n - 2) / (1 - r * r))
        # survival of |t| under normal approx (no scipy); rough but adequate
        p = 1 - math.erf(abs(t) / np.sqrt(2))
    return r, p


def matrices(df):
    n = len(METRICS)
    R = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    V = oriented(df)
    for i in range(n):
        for j in range(n):
            R[i, j], P[i, j] = corr_p(V[METRICS[i]].values, V[METRICS[j]].values)
    return R, P


def group_means(R):
    gs = ["input", "feature", "param"]
    out = {}

    def blk(g1, g2):
        v = [R[i, j] for i in range(len(METRICS)) for j in range(len(METRICS))
             if i < j and SPACE[METRICS[i]] == g1 and SPACE[METRICS[j]] == g2
             and not np.isnan(R[i, j])]
        return (float(np.mean(v)), float(np.mean(np.abs(v))), len(v)) if v else (np.nan, np.nan, 0)
    for g in gs:
        out[f"{g.capitalize()}-{g.capitalize()}"] = blk(g, g)
    for a, b in [("input", "feature"), ("input", "param"), ("feature", "param")]:
        out[f"{a.capitalize()}-{b.capitalize()}"] = blk(a, b)
    return out


def stratified(df):
    """Per-dataset Spearman, averaged via Fisher z (controls dataset-scale confound)."""
    mats = []
    for _, g in df.groupby("Dataset"):
        R, _ = matrices(g)
        mats.append(R)
    z = np.nanmean([np.arctanh(np.clip(R, -0.999, 0.999)) for R in mats], axis=0)
    return np.tanh(z)


def heatmap(R, P, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception:
        print(f"  [skip figure: matplotlib unavailable] {path.name}")
        return
    n = len(METRICS)
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(R, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[m] for m in METRICS], rotation=90, fontsize=8)
    ax.set_yticklabels([LABEL[m] for m in METRICS], fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isnan(R[i, j]):
                continue
            star = "*" if (P is not None and not np.isnan(P[i, j]) and P[i, j] < 0.05) else ""
            ax.text(j, i, f"{R[i, j]:.2f}{star}", ha="center", va="center",
                    fontsize=5.5, color="black")
    # space-block dividers (input 0-9, feature 10-12, param 13-15)
    for b in (9.5, 12.5):
        ax.axhline(b, color="k", lw=1.2); ax.axvline(b, color="k", lw=1.2)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman $\\rho$ (oriented)")
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def agreement_outputs(R_pooled, R_strat, df):
    gp = group_means(R_pooled)
    gs = group_means(R_strat)
    g60 = group_means(matrices(df[df.ASR >= 60])[0])
    rows = []
    for k in gp:
        rows.append({"group": k,
                     "pooled_signed": round(gp[k][0], 3), "pooled_abs": round(gp[k][1], 3),
                     "stratified_signed": round(gs[k][0], 3),
                     "asr60_signed": round(g60[k][0], 3), "n_pairs": gp[k][2]})
    out = pd.DataFrame(rows)
    out.to_csv(ANADIR / "metric_agreement.csv", index=False)
    print(out.to_string(index=False))
    # NOTE: the paper-facing table is hand-maintained inside metric_disagreement.tex
    # (within-dataset + pooled columns). metric_agreement.csv above is its provenance; we
    # deliberately do NOT emit a second standalone table .tex here (avoids a duplicate
    # \label{tab:metric_agreement}).


def main():
    df = pd.read_csv(MATRIX)
    print(f"Loaded {len(df)} configs (scipy={'yes' if HAVE_SCIPY else 'NO -> numpy fallback'})")
    Rp, Pp = matrices(df)
    Rs = stratified(df)
    pd.DataFrame(Rp, index=METRICS, columns=METRICS).round(3).to_csv(ANADIR / "metric_corr_pooled.csv")
    print("\n--- within/cross-space agreement ---")
    agreement_outputs(Rp, Rs, df)
    heatmap(Rp, Pp, "Spearman rank correlation of stealthiness footprints (pooled, N=%d)" % len(df),
            FIGDIR / "metric_rank_correlation.pdf")
    heatmap(Rs, None, "Spearman rank correlation (per-dataset, Fisher-$z$ averaged)",
            FIGDIR / "metric_rank_correlation_stratified.pdf")


if __name__ == "__main__":
    main()
