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
     {seed0 (published), xrun2..xrun5} - per-metric spread, plus whether the
     winner separations survive (adap's SS below every retrained SS, grond's
     TUP/UCLC below every retrained value). Note the semantics: the grond runs
     are fully independent (own trigger + poison indices); the adap runs share
     one regenerated poisoned set, so their spread isolates training seeds.
  5. replicate campaign (target classes): grond and adaptive_patch at targets
     0/1/2 with up to 5 runs per target - per-target run spread, each target's
     mean shift vs the target-0 run spread, and the winner separations checked
     per target class against the matching BackdoorBench rows (seed variants at
     target 0, the single t1/t2 rows at targets 1 and 2).

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
SEED_VARIANTS = ["seed0", "seed1", "seed2", "seed3", "seed4"]
XRUN_VARIANTS = ["seed0", "xrun2", "xrun3", "xrun4", "xrun5"]
# campaign variant -> target class (seed0 doubles as the published target-0 run)
CAMPAIGN_TARGET = {"seed0": 0, "xrun2": 0, "xrun3": 0, "xrun4": 0, "xrun5": 0}
for _t in (1, 2):
    for _n in (1, 2, 3, 4, 5):
        CAMPAIGN_TARGET[f"xt{_t}run{_n}"] = _t
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

    # ---- 3. target-class check (all BackdoorBench attacks) ----------------------
    # Every attack that has at least one target variant is checked, and each metric's
    # target-induced shift is judged against that attack's OWN seed spread: the question
    # is whether changing the target class moves a metric more than simply retraining
    # does. Aggregated per metric across attacks at the end.
    targets = df[df["variant"].isin(["seed0", "t1", "t2"])]
    target_attacks = sorted(targets[targets["variant"] != "seed0"]["attack"].unique())
    if target_attacks:
        print("\n=== Target-class check (target 0 vs 1 vs 2) ===")
        rel_by_metric = {m: [] for m in METRICS}
        within_by_metric = {m: [0, 0] for m in METRICS}   # [within 2x spread, total]
        for attack in target_attacks:
            sub = targets[targets["attack"] == attack]
            if sub["variant"].nunique() < 2:
                continue
            print(f"\n  -- {attack} " + " ".join(f"{v:>9s}" for v in ["target0", "target1", "target2"]))
            for metric in METRICS + PERF:
                vals = sub.set_index("variant")[metric].reindex(["seed0", "t1", "t2"])
                line = " ".join(f"{v:9.4f}" if pd.notna(v) else "      -  " for v in vals)
                note = ""
                if metric in METRICS and pd.notna(vals.get("seed0")) and vals.notna().sum() >= 2:
                    dev = (vals - vals["seed0"]).abs().max()
                    base = abs(vals["seed0"])
                    rel = dev / base if base > 1e-12 else float("nan")
                    if pd.notna(rel):
                        rel_by_metric[metric].append(rel)
                    spread = sdf[(sdf["metric"] == metric) & (sdf["attack"] == attack)]["std"]
                    if len(spread) and pd.notna(spread.iloc[0]) and spread.iloc[0] > 0:
                        ok = dev <= 2 * spread.iloc[0]
                        within_by_metric[metric][0] += int(ok)
                        within_by_metric[metric][1] += 1
                        note = (f"  max |delta| {dev:.4f} vs seed std {spread.iloc[0]:.4f}"
                                f" -> {'within' if ok else 'BEYOND'} 2x seed spread")
                    elif pd.notna(rel):
                        note = f"  max |delta| {dev:.4f} ({rel*100:.1f}% rel)"
                print(f"     {metric:5s} {line}{note}")
                summary.append({"block": "target", "attack": attack, "metric": metric,
                                "mean": "", "std": "", "cv": "",
                                "n_seeds": int(vals.notna().sum())})

        print("\n  -- summary across attacks (target-induced shift) --")
        for metric in METRICS:
            rels = [r for r in rel_by_metric[metric] if pd.notna(r)]
            if not rels:
                continue
            ok, tot = within_by_metric[metric]
            med, mx = float(np.median(rels)), float(np.max(rels))
            print(f"     {metric:5s} median {med*100:5.1f}%  max {mx*100:5.1f}%  "
                  f"within 2x seed spread: {ok}/{tot}")
            summary.append({"block": "target_summary", "attack": "ALL", "metric": metric,
                            "mean": round(med, 4), "std": round(mx, 4),
                            "cv": (f"{ok}/{tot}" if tot else ""), "n_seeds": len(rels)})

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

    # ---- 5. replicate campaign: winners at target classes 0/1/2, up to 5 runs ---
    camp = df[df["variant"].isin(CAMPAIGN_TARGET) & df["attack"].isin(WINNER_ATTACKS)].copy()
    camp["target"] = camp["variant"].map(CAMPAIGN_TARGET)
    if (camp["target"] > 0).any():
        print("\n=== Campaign: grond / adap_patch at targets 0/1/2 (runs per target) ===")
        for attack in WINNER_ATTACKS:
            sub = camp[camp["attack"] == attack]
            if sub.empty:
                continue
            print(f"\n  -- {attack}")
            base_stats = {}
            for metric in METRICS + PERF:
                parts = []
                stats = {}
                for t in (0, 1, 2):
                    got = sub[sub["target"] == t][metric].dropna()
                    if got.empty:
                        parts.append(f"t{t}:        -        ")
                        continue
                    stats[t] = (got.mean(), got.std(ddof=1) if len(got) > 1 else float("nan"),
                                len(got))
                    parts.append(f"t{t}: {got.mean():8.4f}±{got.std():6.4f} n={len(got)}")
                    summary.append({"block": f"campaign_t{t}", "attack": attack,
                                    "metric": metric, "mean": round(got.mean(), 4),
                                    "std": round(got.std(), 4),
                                    "cv": round(got.std() / abs(got.mean()), 4)
                                    if got.mean() != 0 else "", "n_seeds": len(got)})
                note = ""
                if metric in METRICS and 0 in stats:
                    m0, s0, _ = stats[0]
                    base_stats[metric] = stats
                    shifts = [abs(stats[t][0] - m0) for t in (1, 2) if t in stats]
                    if shifts and pd.notna(s0) and s0 > 0:
                        ok = max(shifts) <= 2 * s0
                        note = (f"  max target shift {max(shifts):.4f} vs t0 run std {s0:.4f}"
                                f" -> {'within' if ok else 'BEYOND'} 2x run spread")
                        summary.append({"block": "campaign_shift", "attack": attack,
                                        "metric": metric, "mean": round(max(shifts), 4),
                                        "std": round(s0, 4),
                                        "cv": "within" if ok else "beyond",
                                        "n_seeds": len(shifts) + 1})
                print(f"     {metric:5s} " + "  ".join(parts) + note)

        print("\n  Winner separations per target class (vs the BB rows at that target):")
        for t in (0, 1, 2):
            bb_t = df[df["variant"].isin(SEED_VARIANTS if t == 0 else [f"t{t}"])
                      & df["attack"].isin(BB_ATTACKS)]
            for attack, metric in WINNER_CLAIMS:
                wvals = camp[(camp["attack"] == attack)
                             & (camp["target"] == t)][metric].dropna()
                field = bb_t[metric].dropna()
                if wvals.empty or field.empty:
                    continue
                ok = wvals.max() < field.min()
                print(f"    target {t}  {attack:15s} {metric:5s} runs max {wvals.max():.4f} "
                      f"vs BB min {field.min():.4f} -> winner "
                      f"{'HOLDS' if ok else 'BROKEN'} over {len(wvals)} run(s)")
                summary.append({"block": f"campaign_winner_t{t}", "attack": attack,
                                "metric": metric, "mean": round(float(wvals.max()), 4),
                                "std": round(float(field.min()), 4),
                                "cv": "holds" if ok else "broken",
                                "n_seeds": len(wvals)})

    pd.DataFrame(summary).to_csv(OUT_CSV, index=False)
    print(f"\nsummary -> {OUT_CSV}")


if __name__ == "__main__":
    main()
