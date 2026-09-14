# analysis/

Derived analyses over the paper's per-configuration results. Everything an analysis needs and
produces lives in its own subdirectory; nothing is written outside `analysis/`.

## Shared backbone

- `metrics_matrix.csv` — one row per `(Dataset, Attack, PR_level)` for ResNet18 (76
  configurations): the 16 footprint metrics plus ASR and BA, as published in the paper's tables.
  Consumed by every analysis below. `build_metrics_matrix.py` regenerates it from the LaTeX
  source of the result tables (`../residual_correlations/metric_tables.tex`) and cross-checks it
  against `../dataframing_results_df.csv`.

## Analyses

| Directory | Question | Key output |
|---|---|---|
| `analysis1/` | Do the metrics agree within a space, and across spaces? | `metric_corr_pooled.csv`, `metric_agreement.csv`, `metric_rank_correlation.pdf` (Figure 2, Table 12) |
| `analysis4/` | Do the conclusions survive filtering out low-ASR (failed) attacks? | `asr_sensitivity.csv`, `asr_filter_rank_stability.pdf` (Table 13) |
| `analysis6/` | Do feature-space rankings transfer across architectures? | `arch_stability_feature.csv`, `vgg_newmetrics.csv`, `arch_feature_scatter.pdf` (Figure 5) |
| `analysis7/` | How uncertain are the agreement numbers? | `agreement_bootstrap_ci.csv` — bootstrap 95% CIs (2000 draws), including a cluster bootstrap over whole attacks |
| `analysis8/` | Does CDBI depend on the embedding it is computed in? | `cdbi_sensitivity_summary.csv` — SS/CDBI over raw, PCA and t-SNE embeddings across perplexities and seeds; per-configuration rows in `results/` |
| `analysis9/` | How stable are the model-dependent metrics across training seeds and target classes? | `seed_stability_summary.csv` — recomputed metrics of retrained replicates; per-record rows in `results/` |
| `analysis10/` | Is within-space agreement inflated by metrics derived from the same quantity? | `agreement_excluding_derived.csv` — the agreement gap without PSNR, CDBI and TUP |
| `analysis11/` | Does the UCLC ranking transfer from ResNet18 to VGG16? | `vgg_uclc.csv` — UCLC of the VGG16 checkpoints (models only, data-free) |

The aggregation scripts (`analysis1`, `4`, `6`, `7`, `10`, and the `aggregate_*.py` of `8` and
`9`) read only committed CSVs and run in minutes on a CPU; `../reproduce_analyses.sh` runs them
all. The per-configuration scripts `analysis8/cdbi_embedding_sensitivity.py`,
`analysis9/seed_stability_metrics.py`, `analysis6/analysis6_vgg_newmetrics.py` and
`analysis11/vgg_uclc_extension.py` need the data package (records, saved features, or the
retrained replicates under `large_files/replicates/`) and, for `analysis9`, a GPU; they are run
one configuration per invocation, see their docstrings.
