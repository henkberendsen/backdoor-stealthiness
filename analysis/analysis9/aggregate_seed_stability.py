#!/usr/bin/env python3
"""
Analysis #9 aggregation - seed and target-class stability of the metrics.

Consumes the per-record CSVs written by seed_stability_metrics.py and produces
the reported numbers:

  1. per attack x metric: mean, std and CV (std/|mean|) over training seeds
     {0, 1, 2}, plus the same for BA/ASR;
  2. rank stability: the attack ordering induced by each metric, per seed, and
     Kendall's tau between every seed pair (n=4 attacks, so tau is coarse -
     the per-metric orderings are also printed verbatim);
  3. target-class check: badnet at target 0 (the seed-0 row) versus targets 1
     and 2 - each metric's deviation, compared against the seed spread of the
     same metric so "within seed noise" is a defensible claim;
  4. author-rerun check for the space winners: grond and adaptive_patch over
     {seed0 (published), xrun2, xrun3} - per-metric spread, plus whether the
     winner separations survive (adap's SS below every retrained SS, grond's
     TUP/UCLC below every retrained value). Note the semantics: the grond runs
     are fully independent (own trigger + poison indices); the adap runs share
     one regenerated poisoned set, so their spread isolates training seeds.

Usage:
    python analysis/analysis9/aggregate_seed_stability.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
OUT_CSV = HERE / "seed_stability_summary.csv"

METRICS = ["SS", "CDBI", "DSWD", "UCLC", "TAC", "TUP"]
PERF = ["BA", "ASR"]
SEED_VARIANTS = ["seed0", "seed1", "seed2"]
XRUN_VARIANTS = ["seed0", "xrun2", "xrun3"]
BB_ATTACKS = ["badnet", "blended", "wanet", "bpp"]
WINNER_ATTACKS = ["grond", "adaptive_patch"]
# lower = stealthier for these; the paper's space winners must stay below the field
WINNER_CLAIMS = [("adaptive_patch", "SS"), ("grond", "TUP"), ("grond", "UCLC")]


def load_rows():
    frames = [pd.read_csv(p) for p in sorted(RESULT_DIR.glob("*.csv"))]
    if not frames:
        raise SystemExit(f"no result CSVs under {RESULT_DIR}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["variant", "attack"], keep="last")


def main():
    df = load_rows()
    seeds = df[df["variant"].isin(SEED_VARIANTS) & df["attack"].isin(BB_ATTACKS)]
    attacks = sorted(seeds["attack"].unique())
    summary = []

    # ---- 1. per-attack value stability across seeds -----------------------------
    print("=== Seed stability (CIFAR-10 / ResNet18 / PR 5%) ===")
    for metric in METRICS + PERF:
        print(f"\n--- {metric} ---")
        for attack in attacks:
            vals = (seeds[seeds["attack"] == attack]
                    .set_index("variant")[metric]
                    .reindex(SEED_VARIANTS))
            got = vals.dropna()
            if len(got) < 2:
                print(f"  {attack:8s} only {len(got)} seed(s) so far")
                continue
            cv = got.std() / abs(got.mean()) if got.mean() != 0 else float("nan")
            print(f"  {attack:8s} " +
                  " ".join(f"{v:9.4f}" if pd.notna(v) else "      -  " for v in vals)
                  + f"   mean {got.mean():9.4f}  std {got.std():8.4f}  CV {cv:6.1%}")
            summary.append({"block": "seed", "attack": attack, "metric": metric,
                            "mean": round(got.mean(), 4), "std": round(got.std(), 4),
                            "cv": round(cv, 4), "n_seeds": len(got)})

    # median CV per metric over attacks (the compact [x] for the draft)
    sdf = pd.DataFrame([s for s in summary if s["block"] == "seed"])
    if not sdf.empty:
        print("\nMedian CV across attacks, per metric:")
        for metric in METRICS:
            sub = sdf[sdf["metric"] == metric]
            if not sub.empty:
                print(f"  {metric:5s} {sub['cv'].median():6.1%}")

    # ---- 2. rank stability across seeds ----------------------------------------
    print("\n=== Rank stability across seeds (attack ordering per metric) ===")
    for metric in METRICS:
        pivot = (seeds.pivot_table(index="attack", columns="variant", values=metric)
                 .reindex(attacks))
        have = [v for v in SEED_VARIANTS if v in pivot.columns
                and pivot[v].notna().sum() >= 3]
        if len(have) < 2:
            print(f"  {metric:5s} not enough complete seeds yet")
            continue
        taus = []
        for a, b in itertools.combinations(have, 2):
            sub = pivot[[a, b]].dropna()
            taus.append(kendalltau(sub[a], sub[b])[0])
        orders = {v: "<".join(pivot[v].dropna().sort_values().index) for v in have}
        print(f"  {metric:5s} tau mean {np.mean(taus):+.2f} (min {np.min(taus):+.2f})  "
              + "  ".join(f"{v}: {o}" for v, o in orders.items()))
        summary.append({"block": "rank", "attack": "", "metric": metric,
                        "mean": round(float(np.mean(taus)), 3),
                        "std": round(float(np.min(taus)), 3), "cv": "",
                        "n_seeds": len(have)})

    # ---- 3. target-class check (badnet) -----------------------------------------
    targets = df[df["variant"].isin(["seed0", "t1", "t2"]) & (df["attack"] == "badnet")]
    if len(targets) >= 2:
        print("\n=== Target-class check (badnet: target 0 vs 1 vs 2) ===")
        base = targets[targets["variant"] == "seed0"]
        seed_spread = {m: sdf[(sdf["metric"] == m) & (sdf["attack"] == "badnet")]["std"]
                       for m in METRICS}
        for metric in METRICS + PERF:
            vals = targets.set_index("variant")[metric].reindex(["seed0", "t1", "t2"])
            line = " ".join(f"{v:9.4f}" if pd.notna(v) else "      -  " for v in vals)
            note = ""
            if metric in METRICS and not base.empty:
                spread = seed_spread.get(metric)
                if spread is not None and len(spread) and vals.notna().sum() >= 2:
                    dev = (vals - vals["seed0"]).abs().max()
                    note = (f"  max |delta| {dev:.4f} vs seed std {spread.iloc[0]:.4f}"
                            f" -> {'within' if dev <= 2 * spread.iloc[0] else 'beyond'}"
                            f" 2x seed spread")
            print(f"  {metric:5s} {line}{note}")
            summary.append({"block": "target", "attack": "badnet", "metric": metric,
                            "mean": "", "std": "", "cv": "",
                            "n_seeds": int(vals.notna().sum())})

    # ---- 4. author reruns of the space winners (grond / adaptive_patch) ---------
    xruns = df[df["variant"].isin(XRUN_VARIANTS) & df["attack"].isin(WINNER_ATTACKS)]
    if (xruns["variant"] != "seed0").any():
        print("\n=== Author reruns: grond (independent) / adap_patch (shared poison set) ===")
        for metric in METRICS + PERF:
            for attack in WINNER_ATTACKS:
                vals = (xruns[xruns["attack"] == attack]
                        .set_index("variant")[metric].reindex(XRUN_VARIANTS))
                got = vals.dropna()
                if len(got) < 2:
                    continue
                cv = got.std() / abs(got.mean()) if got.mean() != 0 else float("nan")
                print(f"  {metric:5s} {attack:15s} " +
                      " ".join(f"{v:9.4f}" if pd.notna(v) else "      -  " for v in vals)
                      + f"   mean {got.mean():9.4f}  std {got.std():8.4f}  CV {cv:6.1%}")
                summary.append({"block": "xrun", "attack": attack, "metric": metric,
                                "mean": round(got.mean(), 4), "std": round(got.std(), 4),
                                "cv": round(cv, 4), "n_seeds": len(got)})

        print("\n  Winner separations vs ALL retrained BackdoorBench models:")
        bb_rows = df[df["variant"].isin(SEED_VARIANTS) & df["attack"].isin(BB_ATTACKS)]
        for attack, metric in WINNER_CLAIMS:
            wvals = xruns[xruns["attack"] == attack][metric].dropna()
            field_min = bb_rows[metric].min()
            ok = wvals.max() < field_min
            print(f"    {attack:15s} {metric:5s} runs max {wvals.max():.4f} vs BB field min "
                  f"{field_min:.4f} -> winner {'HOLDS' if ok else 'BROKEN'} "
                  f"over {len(wvals)} run(s)")
            summary.append({"block": "winner", "attack": attack, "metric": metric,
                            "mean": round(float(wvals.max()), 4),
                            "std": round(float(field_min), 4),
                            "cv": "holds" if ok else "broken", "n_seeds": len(wvals)})

    pd.DataFrame(summary).to_csv(OUT_CSV, index=False)
    print(f"\nsummary -> {OUT_CSV}")


if __name__ == "__main__":
    main()
