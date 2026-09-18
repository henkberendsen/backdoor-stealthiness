#!/usr/bin/env python3
"""
Evaluate one trained record in all three observation spaces and append a run row.

A record is the output of an attack implementation (a backdoored model plus its poisoned data),
stored as large_files/record/<attack>_<arch>_<dataset>_p<rate>/ (pNone for DFBA, which poisons no
training data; prototype_<arch>_<dataset>_pNone is the benign model). For the requested record the
script

  performance  benign accuracy on the full clean test set and attack success rate on the poisoned
               test set (target-class samples excluded, as in the paper);
  input        l1, l2, linf, MSE, PSNR, SSIM, LPIPS, IS, pHash and SAM between the clean test
               images and their triggered counterparts, on raw [0, 1] pixels;
  feature      SS and CDBI on the fixed t-SNE embedding (eval_utils.TSNE_KWARGS) of the
               penultimate-layer features of the target-class and poisoned training samples, and
               DSWD from the clean/triggered test features;
  parameter    UCLC from the weights, and TAC and TUP from the activation differences between
               clean and triggered test images.

Intermediate artifacts (features, predictions, activation differences, t-SNE embeddings) are read
from large_files/ when present and extracted otherwise, so the published intermediates reproduce
the paper's tables while a new record is evaluated from scratch. Feature extraction uses the
deterministic test transform (ToTensor + dataset normalization).

    python scripts/evaluate_record.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05
    python scripts/evaluate_record.py --attack dfba --dataset cifar10 --spaces performance parameter
    python scripts/evaluate_record.py --attack blended --dataset cifar10 --poison_rate 0.05 \
        --sample_size 500                       # CPU: subsample the input-space image pairs

The result is one row appended to results/runs.csv: configuration, BA, ASR, the sixteen footprint
metrics, the t-SNE setting, and timing. A GPU is recommended; on a CPU the input-space metrics
(LPIPS and IS run neural networks) are the slow part, hence --sample_size.
"""
import argparse
import copy
import csv
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)          # the record loaders resolve the attack submodules from the cwd
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from torchvision import transforms                    # noqa: E402
from eval_utils import (                              # noqa: E402
    CDBI, DSWD, IMG_SIZE_DICT, IS, LPIPS, MSE, PSNR, SAM, SS, SSIM, TAC, TSNE_KWARGS, TUP, UCLC,
    detect_image_size_from_attack_path, experiment_variable_identifier, extract_preds,
    filter_target_class, get_dataset, l1_distance, l2_distance, linf_distance, load_adap,
    load_backdoorbench, load_dfba, load_dfst, load_grond, load_model_state, metric_over_batches,
    pHash, save_tac_activations, save_test_feature_space, save_train_feature_space, save_tsne,
)

NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    "tiny": ([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    "imagenette": ([0.4671, 0.4593, 0.4306], [0.2692, 0.2657, 0.2884]),
}
BACKDOORBENCH = ["badnet", "blended", "wanet", "bpp", "narcissus"]
ADAPTIVE = ["adaptive_patch", "adaptive_blend"]
ATTACKS = BACKDOORBENCH + ADAPTIVE + ["dfst", "dfba", "grond", "prototype"]
SPACES = ["performance", "input", "feature", "parameter"]
INPUT_METRICS = {"l1": l1_distance, "l2": l2_distance, "linf": linf_distance, "MSE": MSE,
                 "PSNR": PSNR, "SSIM": SSIM, "LPIPS": LPIPS, "IS": IS, "pHash": pHash, "SAM": SAM}
FIELDS = (["timestamp", "model", "dataset", "attack", "poison_rate", "target_class", "train_seed",
           "BA", "ASR"] + list(INPUT_METRICS) + ["SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP",
           "n_input_pairs", "tsne_seed", "tsne_perplexity", "device", "seconds"])


def record_name(attack, model, dataset, poison_rate):
    rate = None if attack in ("dfba", "prototype") else poison_rate
    return f"{attack}_{experiment_variable_identifier(model, dataset, rate)}"


def atk_id(attack, poison_rate):
    """Name of the intermediate files of a configuration, e.g. badnet_p0.05 or dfba."""
    return attack if attack in ("dfba", "prototype") else f"{attack}_p{poison_rate}"


def train_seed(record_dir):
    """Training seed stored by BackdoorBench (info.pickle); other implementations store none."""
    info = record_dir / "info.pickle"
    if not info.exists():
        return ""
    try:
        try:
            d = torch.load(info, weights_only=False)
        except Exception:
            with open(info, "rb") as f:
                d = pickle.load(f)
        d = d if isinstance(d, dict) else vars(d)
        return d.get("random_seed", "")
    except Exception:
        return ""


def load_clean(dataset, model, dirs, transform, target_class, img_size):
    """Clean datasets under one transform (test sets without the target class) + benign model."""
    record = {}
    for key in ["train", "test", "train_transformed", "test_transformed"]:
        ds = get_dataset(dataset, train="train" in key, transforms=transform,
                         data_dir=str(dirs["data"]), img_size=img_size)
        if key.startswith("test"):
            ds = filter_target_class(ds, target_class)
        record[key] = ds
    proto = dirs["record"] / f"prototype_{experiment_variable_identifier(model, dataset, None)}" / "clean_model.pth"
    state = torch.load(proto, weights_only=False, map_location="cpu")
    record["model"] = load_model_state(model, dataset, state.get("model", state) if isinstance(state, dict) and "model" in state else state)
    return record


def load_backdoor(attack, path, dataset, model, clean_record, transform, target_class, data_dir):
    """The attack record with every split under `transform` (the loaders differ in how they take it)."""
    transform_dict = {f"{dataset}_{k}": transform
                      for k in ["train", "test", "train_transformed", "test_transformed"]}
    if attack in BACKDOORBENCH:
        return load_backdoorbench(attack, str(path), dataset, model, transform_dict=transform_dict,
                                  target_class=target_class)
    if attack in ADAPTIVE:
        return load_adap(str(path), dataset, model, clean_record, target_class=target_class)
    if attack == "dfst":
        return load_dfst(str(path), dataset, model, clean_record)
    if attack == "dfba":
        return load_dfba(str(path), dataset, model, clean_record)
    if attack == "grond":
        return load_grond(str(path), dataset, model, transform_dict=transform_dict,
                          target_class=target_class, data_dir=data_dir)
    raise ValueError(attack)


def subset(dataset, n, seed=0):
    """The first `n` indices in a fixed random order (or everything when n is None)."""
    if n is None or n >= len(dataset):
        return dataset, len(dataset)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    return torch.utils.data.Subset(dataset, idx), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", required=True, choices=ATTACKS)
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "vgg16"])
    ap.add_argument("--dataset", default="cifar10", choices=sorted(NORMALIZATION))
    ap.add_argument("--poison_rate", type=float, default=None,
                    help="poisoning rate of the record (omit for dfba and prototype)")
    ap.add_argument("--target_class", type=int, default=0)
    ap.add_argument("--spaces", nargs="+", default=SPACES, choices=SPACES)
    ap.add_argument("--sample_size", type=int, default=None,
                    help="number of clean/triggered test pairs for the input-space metrics (default: all)")
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--large_files", default=os.environ.get("BACKDOOR_STEALTHINESS_DATA",
                                                             str(REPO_ROOT / "large_files")),
                    help="directory with record/, data/ and the intermediate artifacts")
    ap.add_argument("--intermediates", default=None,
                    help="directory for feature_space_*/, tac_activations/, tsne/ and predictions "
                         "(default: the --large_files directory; use another one with --recompute to "
                         "leave the published intermediates untouched)")
    ap.add_argument("--recompute", action="store_true",
                    help="re-extract features, predictions, activation differences and the t-SNE "
                         "embedding even when saved versions exist (they are overwritten)")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "runs.csv"))
    args = ap.parse_args()

    if args.attack not in ("dfba", "prototype") and args.poison_rate is None:
        ap.error(f"--poison_rate is required for {args.attack}")
    if args.attack == "prototype":
        args.spaces = [s for s in args.spaces if s in ("performance", "parameter")]

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and "input" in args.spaces and args.sample_size is None:
        print("note: no GPU found; the input-space metrics over the full test set will be slow "
              "(--sample_size limits the number of image pairs)")

    large = Path(args.large_files)
    inter = Path(args.intermediates) if args.intermediates else large
    dirs = {"record": large / "record", "data": large / "data"}
    dirs.update({name: inter / name for name in ["feature_space_train", "feature_space_test",
                                                  "tac_activations", "tsne",
                                                  "predictions_test_all_labels"]})
    name = record_name(args.attack, args.model, args.dataset, args.poison_rate)
    record_dir = dirs["record"] / name
    if not record_dir.is_dir():
        raise SystemExit(f"record not found: {record_dir}")
    exp_id = f"{args.model}_{args.dataset}"
    cfg = atk_id(args.attack, args.poison_rate)
    img_size = detect_image_size_from_attack_path(str(record_dir)) or IMG_SIZE_DICT.get(args.dataset)

    model_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(*NORMALIZATION[args.dataset])])
    pixel_transform = transforms.ToTensor()

    print(f"[{name}] loading on {device} (image size {img_size}) ...")
    clean = load_clean(args.dataset, args.model, dirs, model_transform, args.target_class, img_size)
    if args.attack == "prototype":
        bd = {"model": clean["model"]}
    else:
        bd = load_backdoor(args.attack, record_dir, args.dataset, args.model, clean,
                           model_transform, args.target_class, str(dirs["data"]))
    model_bd = bd["model"]
    row = {k: float("nan") for k in FIELDS}
    row.update({"timestamp": datetime.now().isoformat(timespec="seconds"), "model": args.model,
                "dataset": args.dataset, "attack": args.attack,
                "poison_rate": "" if args.poison_rate is None else args.poison_rate,
                "target_class": args.target_class, "train_seed": train_seed(record_dir),
                "n_input_pairs": "", "tsne_seed": TSNE_KWARGS["random_state"],
                "tsne_perplexity": TSNE_KWARGS["perplexity"], "device": device.type})

    if "performance" in args.spaces:
        print("performance ...")
        full_test = get_dataset(args.dataset, train=False, transforms=model_transform,
                                data_dir=str(dirs["data"]), img_size=img_size)
        preds_file = dirs["predictions_test_all_labels"] / exp_id / f"{cfg}.pt"
        if args.recompute or not preds_file.exists():
            preds = extract_preds(model_bd, full_test, batch_size=args.batch_size)
            preds_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"predictions_clean": preds, "indices": np.arange(len(full_test))}, preds_file)
        saved = torch.load(preds_file, weights_only=False)
        labels = np.array(full_test.targets)[saved["indices"]]
        row["BA"] = float(np.mean(np.asarray(saved["predictions_clean"]) == labels) * 100)
        if args.attack != "prototype":
            preds_bd = extract_preds(model_bd, bd["test"], batch_size=args.batch_size)
            row["ASR"] = float((preds_bd == args.target_class).float().mean().item() * 100)
        print(f"    BA={row['BA']:.2f}  ASR={row['ASR']:.2f}")

    if "input" in args.spaces:
        print("input space ...")
        # raw pixels: reload the clean record and the attack record with ToTensor only
        clean_px_record = load_clean(args.dataset, args.model, dirs, pixel_transform,
                                     args.target_class, img_size)
        clean_px = clean_px_record["test"]
        bd_px = load_backdoor(args.attack, record_dir, args.dataset, args.model, clean_px_record,
                              pixel_transform, args.target_class, str(dirs["data"]))["test"]
        if len(bd_px) != len(clean_px):
            raise SystemExit(f"clean ({len(clean_px)}) and triggered ({len(bd_px)}) test sets differ in size")
        clean_sub, n = subset(clean_px, args.sample_size)
        bd_sub, _ = subset(bd_px, args.sample_size)
        row["n_input_pairs"] = n
        for metric, func in INPUT_METRICS.items():
            row[metric] = float(metric_over_batches(func, clean_sub, bd_sub, batch_size=args.batch_size))
            print(f"    {metric}={row[metric]:.5g}")

    if "feature" in args.spaces:
        print("feature space ...")
        trainset = bd.get("train") or bd.get("train_transformed")
        train_dir = dirs["feature_space_train"] / exp_id
        test_dir = dirs["feature_space_test"] / exp_id
        if args.attack != "dfba":
            if args.recompute or not (train_dir / f"{cfg}.pt").exists():
                save_train_feature_space(f"{cfg}.pt", model_bd, trainset, str(train_dir),
                                         target_class=args.target_class, batch_size=args.batch_size)
            embedding = dirs["tsne"] / exp_id / cfg / "embedding.pt"
            if args.recompute or not embedding.exists():
                save_tsne(cfg, trainset, str(train_dir), str(dirs["tsne"] / exp_id))
            tsne = torch.load(embedding, weights_only=False)
            benign, poisoned = tsne["features_benign"], tsne["features_poisoned"]
            if len(poisoned):
                row["SS"] = float(SS(benign, poisoned))
                row["CDBI"] = float(CDBI(benign, poisoned, tsne["gt_labels_poisoned"],
                                         len(trainset.classes))[0])
        if args.recompute or not (test_dir / f"{cfg}.pt").exists():
            save_test_feature_space(f"{cfg}.pt", model_bd, clean["test"], bd["test"], str(test_dir),
                                    batch_size=args.batch_size)
        row["DSWD"] = float(DSWD(model_bd, str(test_dir / f"{cfg}.pt"), model_arch=args.model))
        print(f"    SS={row['SS']:.4f}  CDBI={row['CDBI']:.4f}  DSWD={row['DSWD']:.4f}")

    if "parameter" in args.spaces:
        print("parameter space ...")
        row["UCLC"] = UCLC(model_bd).max().item()
        if args.attack != "prototype":
            tac_dir = dirs["tac_activations"] / exp_id
            if args.recompute or not (tac_dir / f"{cfg}.pt").exists():
                save_tac_activations(f"{cfg}.pt", clean["model"], model_bd, clean["test"], bd["test"],
                                     str(tac_dir), batch_size=args.batch_size)
            row["TAC"] = TAC(str(tac_dir / f"{cfg}.pt"), target_class=args.target_class).max().item()
            row["TUP"] = float(TUP(str(tac_dir / f"{cfg}.pt"), model_bd, args.model))
        print(f"    UCLC={row['UCLC']:.4f}  TAC={row['TAC']:.4f}  TUP={row['TUP']:.4f}")

    row["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: (round(v, 6) if isinstance(v, float) and not np.isnan(v) else
                             ("" if isinstance(v, float) else v)) for k, v in row.items()})
    print(f"done in {row['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
