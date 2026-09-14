# Quiet Triggers, Loud Footprints

Code and data for *Quiet Triggers, Loud Footprints: A Tri-Space Measurement Study of Backdoor
Stealthiness* (IEEE Symposium on Security and Privacy, 2027).

The paper measures how detectable backdoor attacks on image classifiers are in three observation
spaces, **input** (the triggered image), **feature** (the penultimate-layer representation) and
**parameter** (the weights and activations), and shows that evidence of stealthiness in one space
does not imply stealthiness in another. This repository holds the measurement library, the
evaluation drivers, the defenses, the derived analyses, and every per-configuration result behind
the paper's tables. The trained models, poisoned datasets and saved intermediates are distributed
separately as a data package (see [Data](#data)).

| | |
|---|---|
| **Architectures** | ResNet18 (all spaces), VGG16 (feature/parameter subset), ViT-Small (feature space, tables only) |
| **Datasets** | CIFAR-10, CIFAR-100, Imagenette (80x80), Tiny-ImageNet (tables only, see [Scope](#scope-of-the-data-package)) |
| **Attacks** | BadNets, Blend, WaNet, BppAttack, Adap-Patch, Adap-Blend, DFST, Narcissus, Grond, DFBA |
| **Input-space metrics** | l1, l2, l-inf, MSE, PSNR, SSIM, LPIPS, IS, pHash, SAM |
| **Feature-space metrics** | SS, DSWD, **CDBI** |
| **Parameter-space metrics** | UCLC, TAC, **TUP** |

CDBI and TUP are introduced by the paper. CDBI measures class-conditional overlap between poisoned
and benign features on a fixed t-SNE embedding (**higher = stealthier**); TUP identifies channels
that are both trigger-responsive and downstream-influential (lower = stealthier).

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Data](#data)
- [Repository layout](#repository-layout)
- [Reproducing the paper](#reproducing-the-paper)
  - [1. Derived analyses from the committed results](#1-derived-analyses-from-the-committed-results-minutes-cpu)
  - [2. Footprint tables from the published intermediates](#2-footprint-tables-from-the-published-intermediates-cpu-about-one-hour-per-dataset)
  - [3. Evaluating a record end to end](#3-evaluating-a-record-end-to-end-gpu-recommended)
  - [4. Defenses](#4-defenses)
  - [5. Training attacks](#5-training-attacks)
- [Extending the benchmark](#extending-the-benchmark)
- [Scope of the data package](#scope-of-the-data-package)
- [Notes on reproducibility](#notes-on-reproducibility)
- [Citation and license](#citation-and-license)

## Requirements

- Linux, Python 3.11 (the reference environment is 3.11.3), `git`.
- The pinned packages in [requirements.txt](requirements.txt): PyTorch 2.5.1, torchvision 0.20.1,
  scikit-learn 1.5.2, scipy 1.14.1, numpy 1.26.4 and the metric libraries (`lpips`, `ImageHash`,
  `torchmetrics`, `POT`, `scikit-image`). Keep scikit-learn at the pinned version when reproducing
  the feature-space tables: SS and CDBI are computed on a t-SNE embedding and t-SNE differs across
  scikit-learn releases.
- A CUDA GPU is recommended for evaluating records and running defenses; the derived analyses
  and the table regeneration in steps 1 and 2 below run on a CPU (step 2 uses many cores if
  available). About 16 GB of RAM.
- Disk: 0.5 GB for the repository and its submodules, about 15 GB for the unpacked data package.
- Internet access on first use: torchvision downloads CIFAR, and the LPIPS and Inception weights
  behind the LPIPS and IS metrics are fetched once into the PyTorch cache. On clusters whose
  compute nodes are offline, run one input-space evaluation on a login node first.

## Setup

```bash
git clone --recurse-submodules https://github.com/henkberendsen/backdoor-stealthiness.git
cd backdoor-stealthiness
python -m venv .venv && source .venv/bin/activate
bash setup.sh            # submodules, loader patch, Python dependencies
bash setup.sh --data     # additionally download and unpack the data package (~10 GB download)
```

`setup.sh` applies [patches/grond_poison_loader.patch](patches/grond_poison_loader.patch) to the
Grond submodule (two small fixes needed to load the published Imagenette records) and, with
`--data`, runs `scripts/download_data.py` followed by `fix_all_backdoorbench_paths.py`, which
rewrites the absolute paths that BackdoorBench stores inside its `attack_result.pt` files.

## Data

Everything that is too large for git lives under `large_files/` and is published as verified
tarballs on Zenodo (DOI `<ZENODO_DOI>`). `python scripts/download_data.py --list` prints the
manifest; components can be fetched individually.

| Component | Contents | Size |
|---|---|---|
| `record/` | 75 trained records: 63 ResNet18 and 12 VGG16 models with their poisoned train/test data (CIFAR-10, CIFAR-100, Imagenette) | 6.0 GB |
| `feature_space_train/` | penultimate-layer features of the target-class and poisoned training samples, per configuration | 0.3 GB |
| `feature_space_test/` | clean and triggered test features plus predictions, per configuration | 2.0 GB |
| `tac_activations/` | activation differences between clean and triggered test images for TAC and TUP | 1.9 GB |
| `predictions_test_all_labels/` | test-set predictions used for benign accuracy | 8 MB |
| `tsne/` | the fixed-seed t-SNE embeddings and plots behind SS and CDBI | 0.1 GB |
| `data/` | the datasets in the layout the loaders expect (CIFAR-10/100 as torchvision folders, Imagenette at 80x80) | 1.0 GB |
| `replicates/` | retrained models for the robustness analyses: five training seeds, two extra target classes, independent Grond/Adap-Patch re-runs, VGG16 checkpoints | 7 GB |

Records are named `<attack>_<arch>_<dataset>_p<rate>` with the decimal point of the poisoning
rate replaced by a dash (`badnet_resnet18_cifar10_p0-05`); DFBA has no poisoning rate and uses
`pNone`, and the benign models are `prototype_<arch>_<dataset>_pNone`.
`experiment_variable_identifier()` in `eval_utils.py` is the single source of truth for this
string. Every loader and script also accepts `--large_files` (or the environment variable
`BACKDOOR_STEALTHINESS_DATA`) to read the package from another location.

## Repository layout

| Path | What it is |
|---|---|
| [eval_utils.py](eval_utils.py) | The measurement library: record loaders for the five attack implementations, the sixteen metrics, feature extraction, and the fixed t-SNE configuration (`TSNE_KWARGS`). Everything else imports this. |
| [scripts/evaluate_record.py](scripts/evaluate_record.py) | Evaluates one record in all three spaces and appends a run row (configuration, BA, ASR, all footprints) to `results/runs.csv`. |
| [scripts/feature_parameter_metrics.py](scripts/feature_parameter_metrics.py) | Regenerates SS, DSWD, CDBI, UCLC, TAC and TUP for one architecture/dataset pair from the saved intermediates. |
| [scripts/download_data.py](scripts/download_data.py) | Fetches and verifies the data package. |
| [defenses/](defenses/) | Standalone defense CLIs: STRIP, Spectral Signatures, SPECTRE, Activation Clustering, I-BAU. Their results are committed under `defenses/results/`. |
| [analysis/](analysis/) | The derived analyses of the paper over the shared `metrics_matrix.csv`; see [analysis/README.md](analysis/README.md). |
| [residual_correlations/](residual_correlations/) | Residual association between footprints and defense outcomes (Figure 3). |
| [tri-space_cdbi.py](tri-space_cdbi.py) | Tri-space summary scores. |
| [reproduce_analyses.sh](reproduce_analyses.sh) | Runs every derived analysis in one go. |
| [eval_metrics.py](eval_metrics.py), [eval.ipynb](eval.ipynb) | The original evaluation driver and its notebook front end, kept for reference; the settings block at the top of the driver selects the configuration. |
| [job_executer.sh](job_executer.sh) | SLURM job template used on our cluster (one job per configuration). The per-experiment job generators are cluster-specific and not part of the artifact. |
| `adap/ backdoorbench/ dfba/ dfst/ grond/` | The attack implementations (git submodules of our forks); each has its own `train.sh`. |
| [fix_all_backdoorbench_paths.py](fix_all_backdoorbench_paths.py), [preprocess_imagenette.py](preprocess_imagenette.py), [tinyimagenet.py](tinyimagenet.py) | Data utilities: record path repair, Imagenette downscaling to 80x80, a Tiny-ImageNet dataset class with a CIFAR-like interface. |

## Reproducing the paper

The paper's results are reproducible at three levels of cost. Level 1 needs only the
repository; levels 2 and 3 need the data package.

### 1. Derived analyses from the committed results (minutes, CPU)

`analysis/metrics_matrix.csv` holds the 76 ResNet18 configurations of the paper (16 footprint
metrics plus BA and ASR); `defenses/results/*.csv` and `dataframing_results_df.csv` hold the
defense outcomes. Every derived analysis reads these files:

```bash
bash reproduce_analyses.sh
```

| Paper element | Script | Output (committed reference) |
|---|---|---|
| Figure 2, Table 12: metric agreement within and across spaces | `analysis/analysis1/analysis1_metric_disagreement.py` | `analysis/analysis1/metric_corr_pooled.csv`, `metric_agreement.csv`, `metric_rank_correlation.pdf` |
| Table 13: sensitivity of the conclusions to ASR filtering | `analysis/analysis4/analysis4_asr_sensitivity.py` | `analysis/analysis4/asr_sensitivity.csv`, `asr_filter_rank_stability.pdf` |
| Figure 5: feature-space rankings across architectures | `analysis/analysis6/analysis6_cross_arch_feature.py` | `analysis/analysis6/arch_stability_feature.csv`, `arch_feature_scatter.pdf` |
| Bootstrap confidence intervals of the agreement gap | `analysis/analysis7/analysis7_uncertainty.py` | `analysis/analysis7/agreement_bootstrap_ci.csv` |
| Agreement without the derived metrics (PSNR, CDBI, TUP) | `analysis/analysis10/analysis10_agreement_excluding_derived.py` | `analysis/analysis10/agreement_excluding_derived.csv` |
| CDBI sensitivity to the embedding (raw, PCA, t-SNE seeds and perplexities) | `analysis/analysis8/aggregate_ranking_agreement.py` | `analysis/analysis8/cdbi_sensitivity_summary.csv` |
| Stability across training seeds and target classes | `analysis/analysis9/aggregate_seed_stability.py` | `analysis/analysis9/seed_stability_summary.csv` |
| Tri-space summary scores | `tri-space_cdbi.py` | `tri_space_metrics_cdbi_{high,low}.csv` |
| Figure 3: residual footprint-defense associations | `residual_correlations/residual_corr.py` | `residuals_heatmap_all.png` |

The scripts rewrite their outputs in place, so `git status` afterwards shows whether anything
differs from the committed reference. The CSV outputs are deterministic (the bootstrap uses a
fixed seed); the PDFs re-render with new metadata but identical content.

### 2. Footprint tables from the published intermediates (CPU, about one hour per dataset)

The model-dependent tables of the paper (Table 5: SS and DSWD; Table 6: UCLC and TAC; Table 10:
CDBI and TUP) are regenerated from the saved features, activation differences and model weights:

```bash
python scripts/feature_parameter_metrics.py --model resnet18 --dataset cifar10
python scripts/feature_parameter_metrics.py --model resnet18 --dataset cifar100
python scripts/feature_parameter_metrics.py --model resnet18 --dataset imagenette
```

Each run writes `results/tables/feature_parameter_resnet18/<dataset>.csv` (one row per
configuration) and stores the t-SNE embeddings under `large_files/tsne/`. Expected outcome: UCLC, TAC, TUP and DSWD reproduce the published values; SS and CDBI depend on the
t-SNE embedding and reproduce the attack rankings, with values that can differ slightly from the
printed ones. The two 2%
CIFAR-10 variants of WaNet and Bpp in the tables have no saved training features and are skipped.

### 3. Evaluating a record end to end (GPU recommended)

`scripts/evaluate_record.py` is the benchmark's per-record driver. It loads a record, extracts
whatever intermediates are missing, computes BA, ASR and all sixteen footprints, and appends one
row to `results/runs.csv`:

```bash
python scripts/evaluate_record.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05
python scripts/evaluate_record.py --attack dfba --model resnet18 --dataset cifar10          # no poisoning rate
python scripts/evaluate_record.py --attack grond --model resnet18 --dataset imagenette --poison_rate 0.05
```

Useful options: `--spaces performance input feature parameter` selects the spaces;
`--sample_size N` limits the number of clean/triggered image pairs for the input-space metrics
(LPIPS and IS run neural networks and are slow on a CPU); `--recompute --intermediates DIR`
re-extracts every intermediate into `DIR` instead of reading the published ones. With the
published intermediates the row for BadNets on CIFAR-10 at 5% is BA 94.5, ASR 100, l1 13.2,
PSNR 25.8, SSIM 0.956, SS 0.507, DSWD 2.01, CDBI 0.574, UCLC 6.59, TAC 3.90, TUP 16.2 (Tables 2,
4, 5, 6 and 10 of the paper).

### 4. Defenses

The five locally implemented defenses take the same arguments and append to
`defenses/results/<defense>.csv`:

```bash
python defenses/strip.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05 --exp_num 1
python defenses/ss.py       ...      # Spectral Signatures
python defenses/spectre.py  ...
python defenses/activation_clustering.py ...
python defenses/ibau.py     ...
```

The committed CSVs are the runs behind the defense tables. CLP and Neural Cleanse were run from
the BackdoorBench submodule (`defense/clp.py`, `defense/nc.py`); FeatureRE, TABOR, BTI-DBF and
BAN were run with their authors' public implementations, and their outcomes are recorded in
`residual_correlations/dataframing_results.py` and `dataframing_results_df.csv`.

### 5. Training attacks

The attacks are trained with the `train.sh` of the corresponding submodule (`backdoorbench` for
BadNets, Blend, WaNet, BppAttack and Narcissus; `adap` for Adap-Patch and Adap-Blend; `dfst`,
`grond`, `dfba`). Each fork documents its own environment; BackdoorBench additionally needs
`kornia` and `pytorch_wavelets`, which are in `requirements.txt`. Training writes a record
directory that the loaders in `eval_utils.py` read directly when it is named as described in
[Data](#data) and placed under `large_files/record/`. The settings of every attack are listed in
the paper's appendix.

## Extending the benchmark

- **A new attack** needs a loader that returns the common record (model, clean and poisoned
  train/test datasets, poisoned-sample indicators, original labels, target class); see the
  `load_*` functions and the `BackdoorDataset` wrappers in `eval_utils.py`. Once it is registered
  in `load_backdoor_record()` and in the driver's attack list, every metric and analysis applies.
- **A new metric** takes either paired image batches (input space), the saved feature files
  (feature space) or the model and activation files (parameter space); add it next to its peers
  in `eval_utils.py` and to the driver's row.
- **A new architecture** needs an entry in `load_model_state()` and, for TAC/UCLC/TUP, the
  layer selection in `TUP()` (the parameter-space metrics are defined for convolutional networks).

## Scope of the data package

- The package covers CIFAR-10, CIFAR-100 and Imagenette for ResNet18, and the DFBA, Grond,
  Narcissus and benign VGG16 models. The Tiny-ImageNet models and the ViT-Small models of the
  paper are not part of the package; their rows are reproducible only at level 1, from the committed tables.
- The published intermediates (features, activation differences, predictions) are the ones the
  paper's tables were computed from. `--recompute` re-extracts them under the deterministic
  test transform, which reproduces DSWD, TAC and TUP closely and the SS/CDBI rankings.

## Notes on reproducibility

- **t-SNE.** SS and CDBI are computed on a two-dimensional t-SNE embedding with perplexity 30,
  PCA initialisation and random state 0 (`TSNE_KWARGS` in `eval_utils.py`). The embedding is
  saved next to the metrics, and the ranking of attacks is insensitive to the seed (analysis 8).
- **Determinism.** All analyses seed their random generators. Inference on a different GPU or
  on a CPU can change the last reported digit of the activation-based metrics (DSWD, TAC, TUP)
  through floating-point summation order; rankings are unaffected.
- **Target class.** All attacks target class 0. Records trained with other target classes are
  evaluated with `--target_class`.
- **Paths.** Nothing is hardcoded to a machine: paths derive from the repository location or from
  `BACKDOOR_STEALTHINESS_DATA` / `BACKDOOR_STEALTHINESS_REPLICATES`.

## Citation and license

```bibtex
@inproceedings{quiet-triggers-2027,
  title     = {Quiet Triggers, Loud Footprints: A Tri-Space Measurement Study of Backdoor Stealthiness},
  author    = {Picek, Stjepan and Xu, Xiaoyun and Berendsen, Henk and Tajalli, Behrad},
  booktitle = {IEEE Symposium on Security and Privacy (SP)},
  year      = {2027}
}
```

The code in this repository is released under the MIT license (see [LICENSE](LICENSE)); the data
package on Zenodo is released under CC BY 4.0. The attack implementations in the submodules keep
the licenses of their original authors.
