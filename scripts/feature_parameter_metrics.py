#!/usr/bin/env python3
"""
Recompute the model-dependent footprint metrics of one architecture/dataset pair from the
saved intermediates, using the fixed t-SNE configuration of eval_utils.TSNE_KWARGS.

For every configuration with saved penultimate-layer training features under
    large_files/feature_space_train/<arch>_<dataset>/<attack>_p<rate>.pt
the script
  1. loads the attack record (needed for the poisoned/cross indicators and the original
     labels of the saved feature subset),
  2. embeds the features with the seeded t-SNE via eval_utils.save_tsne, which stores
     large_files/tsne/<arch>_<dataset>/<attack>_p<rate>/embedding.pt and the scatter plots,
  3. computes SS and CDBI on that embedding, DSWD from the saved test features, and
     UCLC / TAC / TUP from the model weights and the saved activation differences.
DFBA poisons no training data, so it gets DSWD, UCLC, TAC and TUP only.

This regenerates the feature-space table (SS, DSWD), the parameter-space table (UCLC, TAC)
and the CDBI/TUP table of the paper for the given architecture and dataset. It needs no
GPU, but the t-SNE runs take a few minutes per configuration, so run it as a job:

    python scripts/feature_parameter_metrics.py --model resnet18 --dataset cifar10

Output: results/tables/feature_parameter_<arch>/<dataset>.csv (one row per configuration).
Use --large_files to point at a different intermediates directory and --attacks to restrict
the run to a subset of attacks.
"""
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)          # the record loaders resolve the attack submodules from the cwd
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from eval_utils import (                              # noqa: E402
    CDBI,
    DSWD,
    IMG_SIZE_DICT,
    SS,
    TAC,
    TSNE_KWARGS,
    TUP,
    UCLC,
    detect_image_size_from_attack_path,
    experiment_variable_identifier,
    load_backdoor_record,
    load_clean_record,
    save_tsne,
)

N_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenette": 10, "tiny": 200}
TARGET_CLASS = 0

FIELDS = ["model", "dataset", "attack", "poison_rate", "SS", "DSWD", "CDBI", "UCLC", "TAC",
          "TUP", "n_benign", "n_poisoned", "tsne_seed", "tsne_perplexity", "seconds"]


def parse_feature_filename(name):
    """'adaptive_patch_p0.003.pt' -> ('adaptive_patch', 0.003); 'dfba.pt' -> ('dfba', None)."""
    stem = name[:-3] if name.endswith(".pt") else name
    if "_p" not in stem:
        return stem, None
    attack, _, rate = stem.rpartition("_p")
    return attack, float(rate)


def configurations(feature_dir, test_dir, attacks=None):
    """Attack/rate pairs to evaluate: everything with saved train features, plus DFBA when
    its test features exist (it has no train features by construction)."""
    configs = [parse_feature_filename(p.name) for p in sorted(feature_dir.glob("*.pt"))]
    if (test_dir / "dfba.pt").exists() and ("dfba", None) not in configs:
        configs.append(("dfba", None))
    configs = [(a, r) for a, r in configs if a != "prototype"]
    if attacks:
        configs = [(a, r) for a, r in configs if a in attacks]
    return configs


def evaluate(model_arch, dataset, attack, poison_rate, clean_record, dirs, n_classes):
    """All model-dependent footprints of one configuration; NaN where an input is missing."""
    t0 = time.time()
    atk_id = attack if poison_rate is None else f"{attack}_p{poison_rate}"
    exp_id = f"{model_arch}_{dataset}"
    row = {"model": model_arch, "dataset": dataset, "attack": attack,
           "poison_rate": "" if poison_rate is None else poison_rate,
           "tsne_seed": TSNE_KWARGS["random_state"],
           "tsne_perplexity": TSNE_KWARGS["perplexity"]}
    row.update({k: float("nan") for k in ["SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP",
                                          "n_benign", "n_poisoned"]})

    bd_record = load_backdoor_record(dataset=dataset, arch=model_arch, atk=attack,
                                     poison_rate=poison_rate, clean_record=clean_record,
                                     record_dir=str(dirs["record"]))
    model = bd_record["model"]

    # Feature space: seeded t-SNE of the saved training features -> SS and CDBI
    train_features = dirs["feature_space_train"] / exp_id / f"{atk_id}.pt"
    if train_features.exists():
        trainset = bd_record.get("train") or bd_record.get("train_transformed")
        save_tsne(atk_id, trainset, str(dirs["feature_space_train"] / exp_id),
                  str(dirs["tsne"] / exp_id))
        tsne = torch.load(dirs["tsne"] / exp_id / atk_id / "embedding.pt", weights_only=False)
        benign, poisoned = tsne["features_benign"], tsne["features_poisoned"]
        row["n_benign"], row["n_poisoned"] = len(benign), len(poisoned)
        if len(poisoned) > 0:
            row["SS"] = float(SS(benign, poisoned))
            row["CDBI"] = float(CDBI(benign, poisoned, tsne["gt_labels_poisoned"], n_classes)[0])

    test_features = dirs["feature_space_test"] / exp_id / f"{atk_id}.pt"
    if test_features.exists():
        row["DSWD"] = float(DSWD(model, str(test_features), model_arch=model_arch))

    # Parameter space: weights only for UCLC; saved activation differences for TAC and TUP
    row["UCLC"] = UCLC(model).max().item()
    tac_file = dirs["tac_activations"] / exp_id / f"{atk_id}.pt"
    if tac_file.exists():
        row["TAC"] = TAC(str(tac_file), target_class=TARGET_CLASS).max().item()
        row["TUP"] = float(TUP(str(tac_file), model, model_arch))

    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--attacks", nargs="*", default=None,
                    help="restrict to these attacks (default: every saved configuration)")
    ap.add_argument("--large_files", default=str(REPO_ROOT / "large_files"),
                    help="directory holding record/, data/, feature_space_*/, tac_activations/")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: results/tables/feature_parameter_<model>/<dataset>.csv)")
    args = ap.parse_args()

    large = Path(args.large_files)
    dirs = {name: large / name for name in ["record", "data", "feature_space_train",
                                             "feature_space_test", "tac_activations", "tsne"]}
    exp_id = f"{args.model}_{args.dataset}"
    feature_dir = dirs["feature_space_train"] / exp_id
    if not feature_dir.is_dir():
        raise SystemExit(f"no saved training features for {exp_id}: {feature_dir}")
    configs = configurations(feature_dir, dirs["feature_space_test"] / exp_id, args.attacks)
    if not configs:
        raise SystemExit(f"nothing to evaluate for {exp_id}")

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "tables" / f"feature_parameter_{args.model}" / f"{args.dataset}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One clean record per dataset: the image size follows the records (Imagenette is 80x80)
    first_attack, first_rate = configs[0]
    first_path = dirs["record"] / f"{first_attack}_{experiment_variable_identifier(args.model, args.dataset, first_rate)}"
    img_size = detect_image_size_from_attack_path(str(first_path)) or IMG_SIZE_DICT.get(args.dataset)
    print(f"[{exp_id}] {len(configs)} configurations, image size {img_size}, "
          f"t-SNE {TSNE_KWARGS}")
    clean_record = load_clean_record(dataset=args.dataset, arch=args.model,
                                     record_dir=str(dirs["record"]), data_dir=str(dirs["data"]),
                                     target_class=TARGET_CLASS, img_size=img_size)
    n_classes = N_CLASSES[args.dataset]

    failures = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for attack, rate in configs:
            label = attack if rate is None else f"{attack} p{rate}"
            print(f"--- {label}")
            try:
                row = evaluate(args.model, args.dataset, attack, rate, clean_record, dirs, n_classes)
            except Exception as exc:  # keep going; one broken record must not sink the batch
                traceback.print_exc()
                failures.append((label, repr(exc)))
                continue
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
            f.flush()
            print("    " + "  ".join(f"{k}={row[k]:.4f}" for k in ["SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP"]
                                     if not np.isnan(row[k])) + f"  ({row['seconds']:.0f}s)")

    print(f"\nwrote {out_path}")
    if failures:
        print(f"{len(failures)} configuration(s) failed:")
        for label, err in failures:
            print(f"  {label}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
