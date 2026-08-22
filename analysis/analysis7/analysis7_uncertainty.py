#!/usr/bin/env python3
"""
Analysis #7 - Uncertainty of the metric-agreement results.

The correlation/agreement analysis needs uncertainty estimates, and the 76
configurations are not independent (they share attacks/datasets).
We quantify both with bootstrap 95% confidence intervals over the same oriented
Spearman machinery as Analysis #1:

  (a) plain bootstrap over configurations (rows),
  (b) cluster bootstrap over ATTACKS (resample the 10 attacks with replacement and
      keep all configurations of each drawn attack) -- respects within-attack dependence,
  (c) cluster bootstrap over DATASETS (4 clusters; coarse but reported for completeness).

For each draw we recompute the pooled 16x16 Spearman matrix (pairwise-complete, so
DFBA's undefined SS/CDBI are handled) and summarize the six within/cross-space block
means, the within-minus-cross gap, and key individual pairs (SS-CDBI, UCLC-TAC, l1-TUP,
feature-parameter block). The same is repeated on the ASR>=60% subset.

Input : analysis/metrics_matrix.csv  (built by build_metrics_matrix.py)
Output: analysis/analysis7/agreement_bootstrap_ci.csv and printed summary.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent            # analysis/analysis7
ANALYSIS = HERE.parent
MATRIX = ANALYSIS / "metrics_matrix.csv"
OUT = HERE / "agreement_bootstrap_ci.csv"

METRICS = ["l1", "l2", "l_inf", "MSE", "PSNR", "SSIM", "LPIPS", "IS", "pHash", "SAM",
           "SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP"]
SPACE = {**{m: "input" for m in METRICS[:10]},
         **{m: "feature" for m in ["SS", "DSWD", "CDBI"]},
         **{m: "param" for m in ["UCLC", "TAC", "TUP"]}}
INVERT = {"PSNR", "SSIM", "pHash", "CDBI"}        # higher = more stealthy -> flip
N_BOOT = 2000
SEED = 0

BLOCKS = [("input", "input"), ("feature", "feature"), ("param", "param"),
          ("input", "feature"), ("input", "param"), ("feature", "param")]
PAIRS = [("SS", "CDBI"), ("UCLC", "TAC"), ("l1", "TUP"), ("SS", "DSWD")]


def oriented(df):
    X = df[METRICS].astype(float).copy()
    for c in INVERT:
        X[c] = -X[c]
    return X


def block_means(C):
    """C: 16x16 DataFrame of pairwise Spearman rho (pairwise-complete)."""
    out = {}
    for g1, g2 in BLOCKS:
        vals = [C.loc[a, b] for i, a in enumerate(METRICS) for j, b in enumerate(METRICS)
                if i < j and SPACE[a] == g1 and SPACE[b] == g2 and not np.isnan(C.loc[a, b])]
        out[f"{g1}-{g2}"] = np.mean(vals) if vals else np.nan
    within = np.nanmean([out["input-input"], out["feature-feature"], out["param-param"]])
    cross = np.nanmean([out["input-feature"], out["input-param"], out["feature-param"]])
    out["within_minus_cross"] = within - cross
    return out


def stats_for(df):
    C = oriented(df).corr(method="spearman", min_periods=4)
    s = block_means(C)
    for a, b in PAIRS:
        s[f"{a}~{b}"] = C.loc[a, b]
    return s


def bootstrap(df, mode, rng):
    """One bootstrap resample of df under the given clustering mode."""
    if mode == "config":
        return df.sample(n=len(df), replace=True, random_state=rng.integers(1 << 31))
    key = "Attack" if mode == "attack" else "Dataset"
    clusters = df[key].unique()
    drawn = rng.choice(clusters, size=len(clusters), replace=True)
    return pd.concat([df[df[key] == c] for c in drawn], ignore_index=True)


def run(df, label, mode, rng):
    point = stats_for(df)
    draws = []
    for _ in range(N_BOOT):
        try:
            draws.append(stats_for(bootstrap(df, mode, rng)))
        except Exception:
            continue
    dd = pd.DataFrame(draws)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(dd[k], [2.5, 97.5])
        rows.append({"subset": label, "cluster": mode, "quantity": k,
                     "point": round(v, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return rows


def main():
    df = pd.read_csv(MATRIX)
    df_eff = df[df.ASR >= 60].copy()
    rng = np.random.default_rng(SEED)
    all_rows = []
    for label, data in [("all_configs", df), ("asr_ge_60", df_eff)]:
        for mode in ["config", "attack", "dataset"]:
            all_rows.extend(run(data, label, mode, rng))
            print(f"done: {label} / {mode}")
    out = pd.DataFrame(all_rows)
    HERE.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  (N={len(df)} configs, {len(df_eff)} with ASR>=60, "
          f"{N_BOOT} bootstrap draws, seed={SEED})\n")
    key = out[(out.cluster == "attack")]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
