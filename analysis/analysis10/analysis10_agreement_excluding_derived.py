#!/usr/bin/env python3
"""
Analysis #10 - Does the within/cross-space agreement structure survive when the
derived metrics are removed?

Concern: some metrics are constructed from the same underlying quantity
as another metric in the same space, which could inflate within-space agreement.
The derived-by-construction pairs in the pipeline are:

  * PSNR  - a monotone transform of MSE (drop PSNR, keep MSE);
  * CDBI  - computed on the same t-SNE embedding as SS (drop CDBI, keep SS);
  * TUP   - computed from the same TAC activation differences (drop TUP, keep TAC).

This recomputes the Analysis-#1/#7 oriented-Spearman block means on the reduced
13-metric set (9 input / 2 feature / 2 param) next to the full 16-metric set,
with the same bootstrap CI machinery as analysis7 (config bootstrap, and cluster
bootstraps over attacks and datasets), on all configurations and on ASR>=60%.

Caveat to carry into any write-up: with the derived metrics removed, the feature
and parameter within-space blocks each reduce to a single pair (SS-DSWD and
UCLC-TAC), so those "block means" are individual correlations.

Input : analysis/metrics_matrix.csv
Output: analysis/analysis10/agreement_excluding_derived.csv and printed summary.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent            # analysis/analysis10
ANALYSIS = HERE.parent
MATRIX = ANALYSIS / "metrics_matrix.csv"
OUT = HERE / "agreement_excluding_derived.csv"

FULL = ["l1", "l2", "l_inf", "MSE", "PSNR", "SSIM", "LPIPS", "IS", "pHash", "SAM",
        "SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP"]
DERIVED = ["PSNR", "CDBI", "TUP"]
REDUCED = [m for m in FULL if m not in DERIVED]
SPACE = {**{m: "input" for m in FULL[:10]},
         **{m: "feature" for m in ["SS", "DSWD", "CDBI"]},
         **{m: "param" for m in ["UCLC", "TAC", "TUP"]}}
INVERT = {"PSNR", "SSIM", "pHash", "CDBI"}        # higher = more stealthy -> flip
N_BOOT = 2000
SEED = 0
BLOCKS = [("input", "input"), ("feature", "feature"), ("param", "param"),
          ("input", "feature"), ("input", "param"), ("feature", "param")]


def stats_for(df, metrics):
    X = df[metrics].astype(float).copy()
    for c in set(metrics) & INVERT:
        X[c] = -X[c]
    C = X.corr(method="spearman", min_periods=4)
    out = {}
    for g1, g2 in BLOCKS:
        vals = [C.loc[a, b] for i, a in enumerate(metrics) for j, b in enumerate(metrics)
                if i < j and SPACE[a] == g1 and SPACE[b] == g2 and not np.isnan(C.loc[a, b])]
        out[f"{g1}-{g2}"] = np.mean(vals) if vals else np.nan
    within = np.nanmean([out["input-input"], out["feature-feature"], out["param-param"]])
    cross = np.nanmean([out["input-feature"], out["input-param"], out["feature-param"]])
    out["within_minus_cross"] = within - cross
    return out


def bootstrap(df, mode, rng):
    if mode == "config":
        return df.sample(n=len(df), replace=True, random_state=rng.integers(1 << 31))
    key = "Attack" if mode == "attack" else "Dataset"
    clusters = df[key].unique()
    drawn = rng.choice(clusters, size=len(clusters), replace=True)
    return pd.concat([df[df[key] == c] for c in drawn], ignore_index=True)


def run(df, subset, metric_set_name, metrics, mode, rng):
    point = stats_for(df, metrics)
    draws = []
    for _ in range(N_BOOT):
        try:
            draws.append(stats_for(bootstrap(df, mode, rng), metrics))
        except Exception:
            continue
    dd = pd.DataFrame(draws)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(dd[k], [2.5, 97.5])
        rows.append({"subset": subset, "metric_set": metric_set_name, "cluster": mode,
                     "quantity": k, "point": round(v, 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return rows


def main():
    df = pd.read_csv(MATRIX)
    df_eff = df[df.ASR >= 60].copy()
    rng = np.random.default_rng(SEED)
    all_rows = []
    for subset, data in [("all_configs", df), ("asr_ge_60", df_eff)]:
        for name, metrics in [("full16", FULL), ("reduced13", REDUCED)]:
            for mode in ["config", "attack", "dataset"]:
                all_rows.extend(run(data, subset, name, metrics, mode, rng))
                print(f"done: {subset} / {name} / {mode}")
    out = pd.DataFrame(all_rows)
    HERE.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  (N={len(df)} configs, {len(df_eff)} with ASR>=60, "
          f"{N_BOOT} draws, seed={SEED})\n")

    # side-by-side of the headline quantities under the attack-cluster bootstrap
    key = out[out.cluster == "attack"].pivot_table(
        index=["subset", "quantity"], columns="metric_set",
        values=["point", "ci_lo", "ci_hi"], aggfunc="first")
    for subset in ["all_configs", "asr_ge_60"]:
        print(f"--- {subset} (attack-cluster CIs) ---")
        for q in [f"{a}-{b}" for a, b in BLOCKS] + ["within_minus_cross"]:
            r = key.loc[(subset, q)]
            print(f"  {q:20s} full16 {r[('point','full16')]:+.3f} "
                  f"[{r[('ci_lo','full16')]:+.3f},{r[('ci_hi','full16')]:+.3f}]   "
                  f"reduced13 {r[('point','reduced13')]:+.3f} "
                  f"[{r[('ci_lo','reduced13')]:+.3f},{r[('ci_hi','reduced13')]:+.3f}]")
        print()


if __name__ == "__main__":
    main()
