"""
Analysis #11 - extend the VGG16 parameter-space table beyond three attacks.

The tri-space comparison of the paper is instantiated mainly on
ResNet18, and the submitted Table 9 carries VGG16 UCLC/TAC for only three attacks
(Narcissus, Grond, DFBA) - too few, as the paper itself says, for a rank correlation.

UCLC is data-free: it depends only on the convolution and batch-norm weights, so it can
be computed from a bare model checkpoint with no poisoned data, no trigger and no
dataset. That makes it the one parameter-space metric recoverable from the models-only
VGG16 archive (models only), whose records contain a single checkpoint each
and reference their poisoned PNGs by absolute path on the machine that trained them.

TAC, and therefore TUP, are NOT recoverable from those records: both need triggered
inputs. SS, CDBI and DSWD are likewise unavailable. Extending those columns needs the
`bd_train_dataset/` and `bd_test_dataset/` folders (or, for the attacks whose triggers
are deterministic, regeneration of the poisoned data locally).

Correctness gate: the twelve values that appear in the submitted Table 9
(benign / Narcissus / Grond / DFBA on CIFAR-10, CIFAR-100 and Imagenette) must be
reproduced exactly. The script asserts this before reporting the new values.

Usage:  python analysis/analysis11/vgg_uclc_extension.py
Output: analysis/analysis11/vgg_uclc.csv
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
# The VGG16 checkpoints (models only) are read from large_files/replicates/vgg_records/
# by default, or from <BACKDOOR_STEALTHINESS_REPLICATES>/vgg_records.
RECORDS = Path(os.environ.get("BACKDOOR_STEALTHINESS_REPLICATES",
                              REPO / "large_files" / "replicates")) / "vgg_records"
OUT = Path(__file__).resolve().parent / "vgg_uclc.csv"

# values printed in the submitted paper's Table 9; the gate below must reproduce them
PUBLISHED = {
    ("prototype", "cifar10"): 7.75, ("prototype", "cifar100"): 4.63, ("prototype", "imagenette"): 4.66,
    ("narcissus", "cifar10"): 13.5, ("narcissus", "cifar100"): 4.74, ("narcissus", "imagenette"): 6.67,
    ("grond", "cifar10"): 4.12,     ("grond", "cifar100"): 4.77,     ("grond", "imagenette"): 3.78,
    ("dfba", "cifar10"): 20.7,      ("dfba", "cifar100"): 15.7,      ("dfba", "imagenette"): 19.1,
}
ATTACK_ORDER = ["prototype", "badnet", "blended", "wanet", "bpp",
                "adaptive_patch", "adaptive_blend", "narcissus", "grond", "dfba"]
DATASETS = ["cifar10", "cifar100", "imagenette", "tiny"]


def load_state_dict(path):
    """Return the tensor state dict from a record checkpoint, whatever it is wrapped in."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "net", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    return None


def uclc_max(state_dict, normalize=True):
    """Maximum channel Lipschitz upper bound, replicating eval_utils.UCLC().

    eval_utils walks nn.Modules; here we pair each 4-D conv weight with the batch-norm
    that follows it in the state dict, which gives the same conv/BN pairing without
    needing to instantiate the architecture.
    """
    keys = list(state_dict)
    per_layer = []
    for i, k in enumerate(keys):
        if not (k.endswith(".weight") and state_dict[k].dim() == 4):
            continue
        for kk in keys[i + 1: i + 6]:                       # BN follows within a few entries
            if not (kk.endswith(".weight") and state_dict[kk].dim() == 1):
                continue
            base = kk[: -len(".weight")]
            if base + ".running_var" not in state_dict:
                break
            w, gamma = state_dict[k], state_dict[kk]
            std = state_dict[base + ".running_var"].sqrt()
            lips = torch.tensor([
                torch.linalg.svdvals(
                    w[c].reshape(w.shape[1], -1) * (gamma[c] / std[c]).abs()
                ).max()
                for c in range(gamma.shape[0])
            ])
            per_layer.append((lips - lips.mean()) / lips.std() if normalize else lips)
            break
    if not per_layer:
        return None
    return torch.cat(per_layer).max().item()


def main():
    if not RECORDS.is_dir():
        raise SystemExit(f"VGG16 records not found at {RECORDS}")

    values = defaultdict(dict)
    for record in sorted(RECORDS.iterdir()):
        if not record.is_dir() or "_vgg16_" not in record.name:
            continue
        ckpts = sorted(list(record.glob("*.pt")) + list(record.glob("*.pth")))
        if not ckpts:
            print(f"  skip {record.name}: no checkpoint")
            continue
        attack, rest = record.name.split("_vgg16_")
        dataset = rest.rsplit("_p", 1)[0]
        sd = load_state_dict(ckpts[0])
        if sd is None:
            print(f"  skip {record.name}: no state dict inside {ckpts[0].name}")
            continue
        values[dataset][attack] = uclc_max(sd)

    # ---- correctness gate against the submitted table ----------------------------
    failures = []
    for (attack, dataset), want in PUBLISHED.items():
        got = values.get(dataset, {}).get(attack)
        if got is None:
            failures.append(f"{attack}/{dataset}: missing")
        elif abs(got - want) > 0.05:
            failures.append(f"{attack}/{dataset}: got {got:.2f}, Table 9 says {want}")
    if failures:
        raise SystemExit("GATE FAILED - computation does not match the paper:\n  "
                         + "\n  ".join(failures))
    print(f"gate passed: all {len(PUBLISHED)} published Table 9 values reproduced\n")

    header = f"{'attack':18s}" + "".join(f"{d:>13s}" for d in DATASETS)
    print(header)
    print("-" * len(header))
    for attack in ATTACK_ORDER:
        row = f"{attack:18s}"
        for dataset in DATASETS:
            v = values.get(dataset, {}).get(attack)
            mark = "*" if v is not None and (attack, dataset) not in PUBLISHED else " "
            row += f"{v:12.2f}{mark}" if v is not None else f"{'-':>13s}"
        print(row)
    print("\n* = not in the submitted Table 9")

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attack", "dataset", "max_UCLC", "in_submitted_table9"])
        for attack in ATTACK_ORDER:
            for dataset in DATASETS:
                v = values.get(dataset, {}).get(attack)
                if v is not None:
                    w.writerow([attack, dataset, round(v, 4),
                                (attack, dataset) in PUBLISHED])
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
