#!/usr/bin/env python3
"""
Analysis #6(b) - Do the NEW metrics (CDBI, TUP) transfer to VGG16? (S&P, appendix)

The paper computes CDBI/TUP for ResNet18 only. On disk we have VGG16 trained models +
precomputed feature/TAC artifacts for the three attacks Narcissus, Grond, DFBA (the only VGG
attacks trained), on CIFAR-10/CIFAR-100/Imagenette. We recompute CDBI and TUP for VGG16 on
these (no retraining) and compare to the ResNet18 values, as a consistency sanity check.

Coverage: CDBI needs poisoned training data -> Grond, Narcissus only (DFBA is data-free, CDBI
undefined, as on ResNet). TUP -> Grond, Narcissus, DFBA. Too few attacks for a rank
correlation; we report values/orderings.

Runs on a CPU: we patch torch.load to map_location='cpu' (records were saved on GPU) and seed
t-SNE (np.random.seed) for reproducibility.
"""
import sys, os, numpy as np, torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

_orig = torch.load                                        # force CPU load (GPU-saved records)
torch.load = lambda *a, **k: (k.setdefault("map_location", "cpu"), _orig(*a, **k))[1]
import eval_utils as eu
import pandas as pd

# the data package: large_files/ next to the repository, or $BACKDOOR_STEALTHINESS_DATA
LARGE_FILES = Path(os.environ.get("BACKDOOR_STEALTHINESS_DATA", REPO / "large_files"))
RECORD = str(LARGE_FILES / "record")
DATA = str(LARGE_FILES / "data")
FTRAIN = LARGE_FILES / "feature_space_train"
TAC = LARGE_FILES / "tac_activations"
NCLS = {"cifar10": 10, "cifar100": 100, "imagenette": 10}
SEED = 42

# (dataset, high-PR per VGG record); Grond/Narcissus use these, DFBA uses None
PR = {"cifar10": 0.05, "cifar100": 0.007, "imagenette": 0.05}
ATTACKS = ["grond", "narcissus", "dfba"]

# ResNet18 reference (high-PR row) from the verified matrix
m = pd.read_csv(REPO / "analysis" / "metrics_matrix.csv")
def res_ref(ds, atk, col):
    lvl = "none" if atk == "dfba" else "high"
    row = m[(m.Dataset == {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
                            "imagenette": "Imagenette"}[ds]) & (m.Attack == atk) & (m.PR_level == lvl)]
    return float(row[col].iloc[0]) if len(row) and pd.notna(row[col].iloc[0]) else float("nan")

rows = []
for ds in ["cifar10", "cifar100", "imagenette"]:
    try:
        clean = eu.load_clean_record(ds, "vgg16", record_dir=RECORD, data_dir=DATA,
                                     img_size=80 if ds == "imagenette" else None)
    except Exception as e:
        print(f"[skip {ds}: clean record load failed: {e}]"); continue
    for atk in ATTACKS:
        pr = None if atk == "dfba" else PR[ds]
        atk_id = atk if pr is None else f"{atk}_p{pr}"
        try:
            bd = eu.load_backdoor_record(ds, "vgg16", atk, pr, clean, record_dir=RECORD)
        except Exception as e:
            print(f"[skip {atk}/{ds}: bd load failed: {e}]"); continue
        # CDBI (Grond/Narcissus only). create_tsne only uses the trainset's
        # poison_lookup/cross_lookup/original_labels (transform-independent), so either
        # 'train' or 'train_transformed' works.
        cdbi = float("nan")
        trainset = bd.get("train", bd.get("train_transformed"))
        ftrain = FTRAIN / f"vgg16_{ds}" / f"{atk_id}.pt"
        if atk != "dfba" and ftrain.exists() and trainset is not None:
            np.random.seed(SEED)
            fb, fp, gt = eu.create_tsne(trainset, str(ftrain))
            cdbi = float(eu.CDBI(fb, fp, gt, NCLS[ds])[0])
        # TUP (all three)
        tup = float("nan")
        tac = TAC / f"vgg16_{ds}" / f"{atk_id}.pt"
        if tac.exists():
            tup = float(eu.TUP(str(tac), bd["model"], "vgg16"))
        rows.append({"dataset": ds, "attack": atk,
                     "VGG_CDBI": cdbi, "ResNet_CDBI": res_ref(ds, atk, "CDBI"),
                     "VGG_TUP": tup, "ResNet_TUP": res_ref(ds, atk, "TUP")})
        print(f"{ds:11} {atk:9} | CDBI vgg={cdbi:7.3f} res={res_ref(ds,atk,'CDBI'):7.3f} | "
              f"TUP vgg={tup:7.3f} res={res_ref(ds,atk,'TUP'):7.3f}")

out = pd.DataFrame(rows)
out.to_csv(HERE / "vgg_newmetrics.csv", index=False)

# NOTE: the paper-facing table (tab:vgg_newmetrics) is hand-maintained inside cross_arch.tex;
# vgg_newmetrics.csv above is its provenance. No standalone table .tex is emitted.
print("\nwrote", HERE / "vgg_newmetrics.csv")
