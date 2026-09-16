#!/usr/bin/env python3
"""
Analysis #9 - How stable are the model-dependent metrics across training runs?

Computes the feature-space (SS, CDBI, DSWD) and parameter-space (UCLC, TAC, TUP)
metrics for one retrained record, so that seed-1/seed-2 replicates and the
target-class-1/2 variants can be compared against the published seed-0 models.

Protocol choices, applied identically to every variant INCLUDING the seed-0
baseline (which is recomputed here rather than copied from the paper tables):
  * all features are extracted under the deterministic test transform (ToTensor +
    the training normalization) - no augmentation, so run-to-run differences
    measure training stochasticity only;
  * SS and CDBI are computed on a seeded t-SNE (perplexity 30, random_state 0)
    of the penultimate train features - analysis #8 showed rankings are
    insensitive to this seed (rho >= 0.98 on CIFAR-10);
  * TAC/TUP use the benign reference model trained with the SAME seed
    (record_seeds/seedN/prototype_*), and the published prototype for the
    seed-0 and target-class variants (which were trained with seed 0);
  * BA/ASR are taken from the final row of the record's own attack_df.csv.

The recomputed SS/CDBI use the seeded embedding; UCLC, TAC and DSWD serve as
the correctness check for the first validation job.

One record per invocation (a GPU job), appending one row to its own CSV:

    python analysis/analysis9/seed_stability_metrics.py \
        --variant seed0 --attack badnet

Variants: seed0 (published records), seed1/seed2 (scratch replicates),
t1/t2 (target-class variants; badnet only, trained with seed 0),
xrun2/xrun3 (independent grond / adaptive_patch reruns, stored flat under
record_reruns/ with a _runN dir suffix).

grond / adaptive_patch specifics:
  * record markers differ (checkpoint.pth / model.pt instead of
    attack_result.pt) and loading goes through load_grond / load_adap;
  * grond has no untransformed "train" key - its "train_transformed" is used,
    which under this protocol carries the same deterministic test transform;
  * TAC/TUP compare against the PUBLISHED prototype for every variant (there
    are no seed-matched prototypes for those reruns; holding the
    reference fixed isolates attack-model variance);
  * the records store no attack_df.csv, so BA/ASR are measured directly:
    BA on the full clean test set, ASR on the poisoned test set (target class
    excluded, as everywhere in the paper).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)          # eval_utils resolves ./backdoorbench relative to cwd
sys.path.insert(0, str(REPO_ROOT))

import torch                                          # noqa: E402
from torchvision import transforms                    # noqa: E402
from eval_utils import (                              # noqa: E402
    experiment_variable_identifier,
    filter_target_class,
    get_dataset,
    load_adap,
    load_backdoorbench,
    load_grond,
    load_model_state,
    save_train_feature_space,
    save_test_feature_space,
    save_tac_activations,
    SS,
    CDBI,
    DSWD,
    UCLC,
    TAC,
    TUP,
)

# Retrained replicate records (record_seeds/, record_targets/, record_reruns/) are read from
# large_files/replicates/ by default, or from the directory named by the
# BACKDOOR_STEALTHINESS_REPLICATES variable.
SCRATCH = Path(os.environ.get("BACKDOOR_STEALTHINESS_REPLICATES",
                              REPO_ROOT / "large_files" / "replicates"))
PUBLISHED_RECORDS = REPO_ROOT / "large_files" / "record"
INTERMEDIATE = SCRATCH / "step3_intermediate"
OUT_DIR = Path(__file__).resolve().parent / "results"

NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
}
N_CLASSES = {"cifar10": 10}
TSNE_PERPLEXITY = 30.0
TSNE_SEED = 0

# variant -> (backdoor record dir, clean/prototype record dir, target class)
VARIANTS = {
    "seed0": (PUBLISHED_RECORDS, PUBLISHED_RECORDS, 0),
    "seed1": (SCRATCH / "record_seeds" / "seed1", SCRATCH / "record_seeds" / "seed1", 0),
    "seed2": (SCRATCH / "record_seeds" / "seed2", SCRATCH / "record_seeds" / "seed2", 0),
    "seed3": (SCRATCH / "record_seeds" / "seed3", SCRATCH / "record_seeds" / "seed3", 0),
    "seed4": (SCRATCH / "record_seeds" / "seed4", SCRATCH / "record_seeds" / "seed4", 0),
    # target variants are trained with seed 0, so they use the published prototype
    "t1": (SCRATCH / "record_targets" / "t1", PUBLISHED_RECORDS, 1),
    "t2": (SCRATCH / "record_targets" / "t2", PUBLISHED_RECORDS, 2),
}
# the rerun records sit in flat dirs named <attack>_<exp_id>[_targetT]_runN;
# all use the published prototype as the TAC/TUP reference
RECORD_SUFFIX = {}
for _n in (2, 3, 4, 5):
    VARIANTS[f"xrun{_n}"] = (SCRATCH / "record_reruns", PUBLISHED_RECORDS, 0)
    RECORD_SUFFIX[f"xrun{_n}"] = f"_run{_n}"
for _t in (1, 2):
    for _n in (1, 2, 3, 4, 5):
        VARIANTS[f"xt{_t}run{_n}"] = (SCRATCH / "record_reruns", PUBLISHED_RECORDS, _t)
        RECORD_SUFFIX[f"xt{_t}run{_n}"] = f"_target{_t}_run{_n}"
BB_ATTACKS = ["badnet", "blended", "wanet", "bpp"]
MARKERS = {"grond": "checkpoint.pth", "adaptive_patch": "model.pt"}

FIELDS = ["variant", "attack", "dataset", "model", "poison_rate", "target_class",
          "BA", "ASR", "SS", "CDBI", "DSWD", "UCLC", "TAC", "TUP",
          "n_benign", "n_poisoned", "seconds"]


def load_ba_asr(record_path):
    """Final-epoch test accuracy and ASR from the trainer's own attack_df.csv."""
    df_path = record_path / "attack_df.csv"
    if not df_path.exists():
        return float("nan"), float("nan")
    last = pd.read_csv(df_path).iloc[-1]
    return float(last.get("test_acc", float("nan"))), float(last.get("test_asr", float("nan")))


def measure_accuracy(model, dataset):
    """Fraction of samples whose prediction matches the dataset's own labels.

    BA when given the clean test set; ASR when given a poisoned test set whose
    labels are all the target class (grond/adap records ship no attack_df.csv).
    """
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--attack", required=True,
                    choices=BB_ATTACKS + ["grond", "adaptive_patch"])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--poison_rate", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    bd_root, clean_root, target_class = VARIANTS[args.variant]
    exp_id = experiment_variable_identifier(args.model, args.dataset, args.poison_rate)
    record_path = bd_root / f"{args.attack}_{exp_id}{RECORD_SUFFIX.get(args.variant, '')}"
    marker = MARKERS.get(args.attack, "attack_result.pt")
    if not (record_path / marker).exists():
        raise SystemExit(f"missing record: {record_path}")

    atk_id = f"{args.attack}_p{args.poison_rate}.pt"
    inter = INTERMEDIATE / args.variant / f"{args.model}_{args.dataset}"
    feat_train_dir = inter / "feature_space_train"
    feat_test_dir = inter / "feature_space_test"
    tac_dir = inter / "tac_activations"

    # deterministic test transform, used for every split of every variant
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION[args.dataset]),
    ])
    transform_dict = {f"{args.dataset}_{k}": test_t
                      for k in ["train", "test", "train_transformed", "test_transformed"]}

    print(f"[{args.variant}/{args.attack}] loading records "
          f"(target class {target_class}) ...")
    clean_test = get_dataset(args.dataset, train=False, transforms=test_t,
                             data_dir=str(REPO_ROOT / "data"))
    clean_test_no_target = filter_target_class(clean_test, target_class)

    proto_path = clean_root / f"prototype_{args.model}_{args.dataset}_pNone" / "clean_model.pth"
    model_clean = load_model_state(args.model, args.dataset,
                                   torch.load(proto_path, weights_only=False))

    if args.attack in BB_ATTACKS:
        bd_record = load_backdoorbench(args.attack, str(record_path), args.dataset,
                                       args.model, transform_dict=transform_dict,
                                       target_class=target_class)
        trainset_bd = bd_record["train"]
    elif args.attack == "grond":
        # no untransformed "train" key; train_transformed carries the same
        # deterministic transform under this protocol
        bd_record = load_grond(str(record_path), args.dataset, args.model,
                               transform_dict=transform_dict,
                               target_class=target_class)
        trainset_bd = bd_record["train_transformed"]
    else:  # adaptive_patch: poisons are merged into deterministic clean datasets
        clean_dsets = {}
        for key in ["train", "test", "train_transformed", "test_transformed"]:
            ds = get_dataset(args.dataset, train="train" in key, transforms=test_t,
                             data_dir=str(REPO_ROOT / "data"))
            if key.startswith("test"):
                ds = filter_target_class(ds, target_class)
            clean_dsets[key] = ds
        bd_record = load_adap(str(record_path), args.dataset, args.model, clean_dsets,
                              target_class=target_class)
        trainset_bd = bd_record["train"]
    model_bd = bd_record["model"]
    testset_bd = bd_record["test"]

    print("extracting features ...")
    save_train_feature_space(atk_id, model_bd, trainset_bd, str(feat_train_dir),
                             target_class=target_class)
    save_test_feature_space(atk_id, model_bd, clean_test_no_target, testset_bd,
                            str(feat_test_dir))
    save_tac_activations(atk_id, model_clean, model_bd, clean_test_no_target,
                         testset_bd, str(tac_dir))

    print("computing metrics ...")
    train_feats = torch.load(feat_train_dir / atk_id, weights_only=False)
    features = np.asarray(train_feats["features"])
    indices = np.asarray(train_feats["indices"])

    is_poisoned = trainset_bd.poison_lookup[indices]
    is_cross = trainset_bd.cross_lookup[indices]
    is_benign = ~(is_poisoned | is_cross)

    embedded = TSNE(perplexity=TSNE_PERPLEXITY,
                    random_state=TSNE_SEED).fit_transform(features)
    benign = np.concatenate([embedded[is_benign], embedded[is_cross]])
    poisoned = embedded[is_poisoned]
    gt_labels = trainset_bd.original_labels[indices][is_poisoned]

    ss = float(SS(benign, poisoned))
    cdbi, _ = CDBI(benign, poisoned, gt_labels, N_CLASSES[args.dataset])
    dswd = float(DSWD(model_bd, str(feat_test_dir / atk_id), model_arch=args.model))
    uclc = UCLC(model_bd).max().item()
    tac = TAC(str(tac_dir / atk_id)).max().item()
    tup = float(TUP(str(tac_dir / atk_id), model_bd, args.model))
    ba, asr = load_ba_asr(record_path)
    if np.isnan(ba):  # grond/adaptive_patch: no attack_df.csv -> measure directly
        ba = measure_accuracy(model_bd, clean_test)
        asr = measure_accuracy(model_bd, testset_bd)

    out_path = Path(args.out) if args.out else (
        OUT_DIR / f"{args.variant}_{args.attack}_p{args.poison_rate}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "variant": args.variant, "attack": args.attack, "dataset": args.dataset,
            "model": args.model, "poison_rate": args.poison_rate,
            "target_class": target_class, "BA": round(ba, 4), "ASR": round(asr, 4),
            "SS": round(ss, 6), "CDBI": round(float(cdbi), 6),
            "DSWD": round(dswd, 6), "UCLC": round(uclc, 6),
            "TAC": round(tac, 6), "TUP": round(tup, 6),
            "n_benign": len(benign), "n_poisoned": len(poisoned),
            "seconds": round(time.time() - t0, 1),
        })

    print(f"[{args.variant}/{args.attack}] SS={ss:+.4f} CDBI={cdbi:.4f} "
          f"DSWD={dswd:.4f} UCLC={uclc:.4f} TAC={tac:.4f} TUP={tup:.4f} "
          f"BA={ba:.4f} ASR={asr:.4f}")
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
