#!/usr/bin/env bash
# Regenerate every derived analysis of the paper from the committed per-configuration results.
# CPU only, a few minutes in total; run from the repository root after `bash setup.sh`.
#
#   bash reproduce_analyses.sh
#
# Each step prints its summary and rewrites its CSV/PDF outputs in place next to the script, so
# `git status` afterwards shows exactly what (if anything) changed relative to the published
# results. The mapping from paper elements to outputs is in the README.
set -euo pipefail
cd "$(dirname "$0")"

run() { echo; echo "================================================================"; echo "== $*"; echo "================================================================"; python "$@"; }

run analysis/analysis1/analysis1_metric_disagreement.py           # Fig. 2, Table 12
run analysis/analysis4/analysis4_asr_sensitivity.py               # Table 13
run analysis/analysis6/analysis6_cross_arch_feature.py            # Fig. 5 (feature-space ranks across architectures)
run analysis/analysis7/analysis7_uncertainty.py                   # bootstrap confidence intervals of the agreement gap
run analysis/analysis10/analysis10_agreement_excluding_derived.py # agreement without PSNR, CDBI, TUP
run analysis/analysis8/aggregate_ranking_agreement.py             # CDBI embedding sensitivity (aggregation)
run analysis/analysis9/aggregate_seed_stability.py                # seed / target-class stability (aggregation)
run residual_correlations/residual_corr.py                        # Fig. 3 (residual defense-metric associations)

echo
echo "All analyses finished."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Changed files, if any:"
    git status --short -- analysis residual_correlations residuals_heatmap_all.png || true
else
    echo "(plain source tree: compare the regenerated CSVs with the ones in the archive if needed)"
fi
