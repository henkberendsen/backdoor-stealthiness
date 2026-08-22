#!/usr/bin/env python3
"""
Analysis #8 - Does CDBI depend on the embedding it is computed in?

CDBI (and SS) are computed on a 2-D t-SNE embedding of penultimate-layer features.
Both are distance-based, so one can reasonably ask whether the resulting
attack ranking reflects the learned representation or the projection used to
visualise it. This script recomputes both metrics for one attack configuration
under a grid of embeddings:

    raw512          the penultimate features themselves, no projection
    pca50 / pca2    linear projections (pca2 isolates "2-D" from "t-SNE")
    tsne_p{P}_s{S}  perplexity P in {15, 30, 50} x random_state S in {0..4}

The paper uses t-SNE with perplexity 30. Its published CDBI values came from
embeddings that were computed without a fixed random_state and are not stored in
this repository, so this script does not attempt to reproduce them bit-for-bit;
it measures how much the metric and, more importantly, the attack *ranking* move
across embeddings.

Runs on CPU only: it consumes the penultimate features saved under
large_files/feature_space_train/ and never touches a GPU.

One configuration per invocation (mirrors the defense scripts), appending rows to
a shared CSV so the configurations can be run as independent jobs:

    python analysis/analysis8/cdbi_embedding_sensitivity.py \
        --attack badnet --dataset cifar10 --model resnet18 --poison_rate 0.05
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_utils import (  # noqa: E402
    load_clean_record,
    load_backdoor_record,
    experiment_variable_identifier,
    detect_image_size_from_attack_path,
    clustering_score,
    CDBI,
    SS,
    IMG_SIZE_DICT,
)

RECORD_DIR = REPO_ROOT / "large_files" / "record"
DATA_DIR = REPO_ROOT / "large_files" / "data"
FEATURE_DIR = REPO_ROOT / "large_files" / "feature_space_train"
OUT_CSV = Path(__file__).resolve().parent / "cdbi_embedding_sensitivity.csv"
TARGET_CLASS = 0

PERPLEXITIES = [15.0, 30.0, 50.0]     # the paper uses 30
TSNE_SEEDS = [0, 1, 2, 3, 4]

FIELDS = ["dataset", "model", "attack", "poison_rate", "embedding", "method",
          "perplexity", "seed", "dim", "SS", "CDBI", "n_benign", "n_poisoned",
          "n_source_classes", "seconds"]


def load_split_labels(dataset, model, attack, poison_rate):
    """Poison/cross masks and original labels for the saved feature subset.

    Mirrors create_tsne(): 'cross' samples (WaNet/Bpp noise mode) count as benign,
    and poisoned samples keep their ORIGINAL class so CDBI can group by source class.
    """
    exp_id = experiment_variable_identifier(model, dataset, poison_rate)
    attack_path = str(RECORD_DIR / f"{attack}_{exp_id}")
    img_size = detect_image_size_from_attack_path(attack_path) or IMG_SIZE_DICT.get(dataset)

    clean_record = load_clean_record(
        dataset=dataset, arch=model, record_dir=str(RECORD_DIR),
        data_dir=str(DATA_DIR), target_class=TARGET_CLASS, img_size=img_size,
    )
    bd_record = load_backdoor_record(
        dataset=dataset, arch=model, atk=attack, poison_rate=poison_rate,
        clean_record=clean_record, record_dir=str(RECORD_DIR),
    )
    trainset = bd_record.get("train") or bd_record.get("train_transformed")
    return trainset


def load_features(dataset, model, attack, poison_rate):
    """Penultimate features saved for the target-class + poisoned samples."""
    import torch
    exp_dir = FEATURE_DIR / f"{model}_{dataset}"
    path = exp_dir / f"{attack}_p{poison_rate}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no saved features at {path}")
    d = torch.load(path, weights_only=False)
    return np.asarray(d["features"]), np.asarray(d["indices"])


def split(embedded, trainset, indices):
    """Split an embedding into (benign+cross, poisoned) and source-class labels."""
    is_poisoned = trainset.poison_lookup[indices]
    is_cross = trainset.cross_lookup[indices]
    is_benign = ~(is_poisoned | is_cross)

    benign = np.concatenate([embedded[is_benign], embedded[is_cross]])
    poisoned = embedded[is_poisoned]
    gt_labels = trainset.original_labels[indices][is_poisoned]
    return benign, poisoned, gt_labels


def embeddings(features):
    """Yield (name, method, perplexity, seed, embedded_features)."""
    yield "raw512", "raw", "", "", features
    yield "pca50", "pca", "", "", PCA(n_components=50, random_state=0).fit_transform(features)
    yield "pca2", "pca", "", "", PCA(n_components=2, random_state=0).fit_transform(features)
    for perp in PERPLEXITIES:
        for seed in TSNE_SEEDS:
            emb = TSNE(perplexity=perp, random_state=seed).fit_transform(features)
            yield f"tsne_p{int(perp)}_s{seed}", "tsne", perp, seed, emb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--poison_rate", type=float, required=True)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    print(f"[{args.attack} p{args.poison_rate} {args.model}/{args.dataset}] loading ...")
    features, indices = load_features(args.dataset, args.model, args.attack, args.poison_rate)
    trainset = load_split_labels(args.dataset, args.model, args.attack, args.poison_rate)
    n_classes = len(trainset.classes)
    print(f"  features {features.shape}, {n_classes} classes")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for name, method, perp, seed, emb in embeddings(features):
            t0 = time.time()
            benign, poisoned, gt_labels = split(emb, trainset, indices)
            if len(poisoned) == 0:
                print(f"  {name}: no poisoned samples, skipped")
                continue

            ss = SS(benign, poisoned)
            cdbi, per_class = CDBI(benign, poisoned, gt_labels, n_classes)
            elapsed = time.time() - t0

            writer.writerow({
                "dataset": args.dataset, "model": args.model, "attack": args.attack,
                "poison_rate": args.poison_rate, "embedding": name, "method": method,
                "perplexity": perp, "seed": seed, "dim": emb.shape[1],
                "SS": round(float(ss), 6), "CDBI": round(float(cdbi), 6),
                "n_benign": len(benign), "n_poisoned": len(poisoned),
                "n_source_classes": len(per_class), "seconds": round(elapsed, 1),
            })
            f.flush()
            print(f"  {name:16s} SS={ss:+.4f}  CDBI={cdbi:.4f}  ({elapsed:.0f}s)")

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
