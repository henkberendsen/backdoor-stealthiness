#!/usr/bin/env python3
"""
Analysis #8 aggregation - is the CDBI attack ranking stable across embeddings?

Consumes the per-configuration CSVs produced by cdbi_embedding_sensitivity.py
(one row per embedding setting: raw512, pca50, pca2, tsne_p{15,30,50}_s{0..4})
and answers, per dataset and per metric (CDBI, and SS as the companion metric
computed on the same embedding):

  1. cross-embedding ranking agreement: mean/min pairwise Spearman rho and
     Kendall tau between the configuration rankings induced by every pair of
     embedding settings;
  2. t-SNE seed stability: pairwise rank agreement across the 5 seeds at each
     perplexity, and per-configuration value spread (std, std/mean) across the
     15 t-SNE runs;
  3. representation sensitivity: agreement of the mean t-SNE(p=30) ranking with
     raw512 / pca50 / pca2;
  4. top-attack stability: the best-ranked attack under every embedding
     (the paper's CDBI verdict is Narcissus);
  5. agreement with the published ranking (analysis/metrics_matrix.csv, parsed
     from the paper tables). On CIFAR-10 the matrix rows for wanet/bpp at the
     low rate hold the paper's Var (2%) values while the saved features come
     from the p0-003 records, so those two configurations are reported both
     included and excluded.

Absolute values are expected to move a lot across embeddings (raw-512 CDBI is
on a different scale than t-SNE CDBI); the paper's claim rests on the RANKING
agreement this script measures.

Usage:
    python analysis/analysis8/aggregate_ranking_agreement.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RESULT_DIR = HERE / "results"
LEGACY_CSV = HERE / "cdbi_embedding_sensitivity.csv"
MATRIX_CSV = REPO_ROOT / "analysis" / "metrics_matrix.csv"
OUT_CSV = HERE / "cdbi_sensitivity_summary.csv"

METRICS = ["CDBI", "SS"]
MATRIX_DATASET = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
                  "imagenette": "Imagenette", "tiny": "Tiny-ImageNet"}
# matrix rows whose paper value comes from a different record than the saved
# features (paper Var = 2% vs features from the 0.3% record)
PR_MISMATCH = {("cifar10", "wanet", 0.003), ("cifar10", "bpp", 0.003)}
# matrix labels the clean-label "high" rate as 0.05 where the actual (and feature-file)
# rate is 0.007 — same records, different label, so join them via an alias
MATRIX_CONFIG_ALIAS = {
    "cifar100": {"narcissus_p0.05": "narcissus_p0.007", "grond_p0.05": "grond_p0.007"},
}


def load_rows():
    frames = [pd.read_csv(p) for p in sorted(RESULT_DIR.glob("*.csv"))]
    if LEGACY_CSV.exists():
        frames.append(pd.read_csv(LEGACY_CSV))
    if not frames:
        raise SystemExit(f"no result CSVs under {RESULT_DIR} (or {LEGACY_CSV})")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["dataset", "model", "attack", "poison_rate", "embedding"], keep="last")
    df["config"] = df["attack"] + "_p" + df["poison_rate"].astype(str)
    return df


def pivot(df, metric):
    """configs x embeddings value table (rows sorted for deterministic output)."""
    return df.pivot_table(index="config", columns="embedding",
                          values=metric).sort_index()


def pairwise_agreement(table, cols):
    rhos, taus = [], []
    for a, b in itertools.combinations(cols, 2):
        sub = table[[a, b]].dropna()
        if len(sub) < 3:
            continue
        rhos.append(spearmanr(sub[a], sub[b])[0])
        taus.append(kendalltau(sub[a], sub[b])[0])
    return np.array(rhos), np.array(taus)


def agreement_with(table, ref_col, cols):
    out = {}
    for c in cols:
        sub = table[[ref_col, c]].dropna()
        if len(sub) >= 3:
            out[c] = spearmanr(sub[ref_col], sub[c])[0]
    return out


def main():
    df = load_rows()
    summary = []

    def record(dataset, metric, quantity, value, n=""):
        summary.append({"dataset": dataset, "metric": metric,
                        "quantity": quantity, "value": round(float(value), 3),
                        "n": n})

    for dataset, ddf in df.groupby("dataset"):
        n_cfg = ddf["config"].nunique()
        embeddings = sorted(ddf["embedding"].unique())
        tsne_cols = [e for e in embeddings if e.startswith("tsne")]
        print(f"\n=== {dataset}: {n_cfg} configurations, "
              f"{len(embeddings)} embedding settings ===")
        if n_cfg < 3:
            print("fewer than 3 configurations - skipping until the sweep lands")
            continue

        for metric in METRICS:
            table = pivot(ddf, metric)
            print(f"\n--- {metric} ---")

            # 1. agreement across ALL embedding settings
            rhos, taus = pairwise_agreement(table, embeddings)
            if len(rhos) == 0:
                print("insufficient overlapping configurations - sweep still running?")
                continue
            print(f"all-pairs ranking agreement: mean rho={rhos.mean():+.3f} "
                  f"(min {rhos.min():+.3f}), mean tau={taus.mean():+.3f} "
                  f"over {len(rhos)} pairs")
            record(dataset, metric, "allpairs_mean_rho", rhos.mean(), len(rhos))
            record(dataset, metric, "allpairs_min_rho", rhos.min(), len(rhos))
            record(dataset, metric, "allpairs_mean_tau", taus.mean(), len(taus))

            # 2. t-SNE seed stability per perplexity + value spread
            for perp in ["15", "30", "50"]:
                cols = [c for c in tsne_cols if c.startswith(f"tsne_p{perp}_")]
                if len(cols) < 2:
                    continue
                rhos_p, _ = pairwise_agreement(table, cols)
                if len(rhos_p) == 0:
                    continue
                print(f"tsne p={perp}: seed-to-seed mean rho={rhos_p.mean():+.3f} "
                      f"(min {rhos_p.min():+.3f})")
                record(dataset, metric, f"tsne_p{perp}_seed_mean_rho", rhos_p.mean(),
                       len(rhos_p))
            tsne_vals = table[tsne_cols]
            rel_spread = (tsne_vals.std(axis=1) / tsne_vals.mean(axis=1).abs())
            print(f"per-config value spread across {len(tsne_cols)} t-SNE runs: "
                  f"median std/|mean| = {rel_spread.median():.3f} "
                  f"(max {rel_spread.max():.3f})")
            record(dataset, metric, "tsne_value_relspread_median",
                   rel_spread.median(), len(tsne_cols))

            # 3. representation sensitivity vs the paper's setting (t-SNE p=30)
            p30 = [c for c in tsne_cols if c.startswith("tsne_p30_")]
            table = table.assign(tsne_p30_meanrank=table[p30].rank().mean(axis=1))
            rep = agreement_with(table, "tsne_p30_meanrank",
                                 ["raw512", "pca50", "pca2"])
            for name, rho in rep.items():
                print(f"tsne_p30 (mean over seeds) vs {name}: rho={rho:+.3f}")
                record(dataset, metric, f"p30_vs_{name}_rho", rho, n_cfg)

            # 4. best-ranked attack under every embedding
            asc = metric != "CDBI"   # higher CDBI = stealthier; lower SS = stealthier
            attacks = ddf["attack"].unique()
            tops = {}
            for e in embeddings:
                ranks = table[e].rank(ascending=asc)
                mean_rank = {a: ranks[[c for c in table.index
                                       if c.startswith(a + "_p")]].mean()
                             for a in attacks}
                tops[e] = min(mean_rank, key=mean_rank.get)
            top_counts = pd.Series(tops).value_counts()
            print(f"top-ranked attack per embedding: "
                  + ", ".join(f"{a} ({c}/{len(embeddings)})"
                              for a, c in top_counts.items()))
            record(dataset, metric, f"top_attack_{top_counts.index[0]}_share",
                   top_counts.iloc[0] / len(embeddings), len(embeddings))

            # 5. agreement with the published (paper-table) ranking
            if MATRIX_CSV.exists() and metric in ("CDBI", "SS"):
                m = pd.read_csv(MATRIX_CSV)
                m = m[m["Dataset"] == MATRIX_DATASET.get(dataset, "")]
                m = m.assign(config=m["Attack"].str.lower() + "_p" + m["PR"].astype(str))
                m["config"] = m["config"].replace(MATRIX_CONFIG_ALIAS.get(dataset, {}))
                pub = m.set_index("config")[metric].astype(float)
                pub = pub[~pub.index.duplicated()]
                joined = table.join(pub.rename("published"), how="inner")
                if len(joined) >= 3:
                    full = agreement_with(joined, "published", embeddings)

                    def mismatched(cfg):
                        attack, _, rate = cfg.rpartition("_p")
                        return (dataset, attack, float(rate)) in PR_MISMATCH

                    keep = [c for c in joined.index if not mismatched(c)]
                    clean = agreement_with(joined.loc[keep], "published", embeddings)
                    if not full or not clean:
                        continue
                    mean_full = np.mean(list(full.values()))
                    mean_clean = np.mean(list(clean.values()))
                    print(f"vs published ranking: mean rho={mean_full:+.3f} "
                          f"(n={len(joined)}); excluding PR-mismatched configs "
                          f"{mean_clean:+.3f} (n={len(keep)})")
                    record(dataset, metric, "vs_published_mean_rho", mean_full,
                           len(joined))
                    record(dataset, metric, "vs_published_clean_mean_rho",
                           mean_clean, len(keep))

    out = pd.DataFrame(summary)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsummary -> {OUT_CSV}")


if __name__ == "__main__":
    main()
