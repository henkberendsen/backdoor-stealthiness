#!/usr/bin/env bash
# One-time setup of the evaluation environment.
#
#   bash setup.sh            # submodules, local loader patch, Python dependencies
#   bash setup.sh --data     # additionally download and unpack the large files (see below)
#
# The large files (trained records, saved features and activations, datasets) are not part of
# the git repository. They are published as tarballs on Zenodo; scripts/download_data.py
# fetches and verifies them and unpacks them under large_files/. Set BACKDOOR_STEALTHINESS_DATA
# to place them elsewhere.
set -euo pipefail
cd "$(dirname "$0")"

echo "== attack implementations (git submodules)"
git submodule update --init --recursive

# Our Grond fork needs two small changes to load the published Imagenette records (the UPGD
# trigger is 80x80, the raw images are not) and to download CIFAR on first use.
PATCH="$PWD/patches/grond_poison_loader.patch"
if git -C grond apply --check "$PATCH" 2>/dev/null; then
    git -C grond apply "$PATCH"
    echo "applied patches/grond_poison_loader.patch"
elif git -C grond apply --check -R "$PATCH" 2>/dev/null; then
    echo "patches/grond_poison_loader.patch already applied"
else
    echo "WARNING: patches/grond_poison_loader.patch does not apply cleanly; check grond/poison_loader.py" >&2
fi

echo "== Python dependencies"
python -m pip install -r requirements.txt

if [[ "${1:-}" == "--data" ]]; then
    echo "== large files"
    python scripts/download_data.py
    # BackdoorBench records store absolute paths from the machine that trained them
    python fix_all_backdoorbench_paths.py
fi

mkdir -p script_logging results/tables
echo "done"
