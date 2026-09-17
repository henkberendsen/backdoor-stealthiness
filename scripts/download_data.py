#!/usr/bin/env python3
"""
Download the large files of the artifact from the permanent repository and unpack them under
large_files/ (or the directory named by BACKDOOR_STEALTHINESS_DATA).

Every tarball is listed in MANIFEST with its sha256 checksum; a file that is already present
with the right checksum is not downloaded again. Components can be selected individually, and
--list prints the manifest without downloading anything:

    python scripts/download_data.py                       # everything
    python scripts/download_data.py --no-replicates       # everything but the retrained replicates
    python scripts/download_data.py record tsne           # selected components
    python scripts/download_data.py --list

Sizes are approximate. The record component holds the trained models together with their
poisoned datasets (many small PNG files), so unpacking it takes a few minutes.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LARGE_FILES = Path(os.environ.get("BACKDOOR_STEALTHINESS_DATA", REPO_ROOT / "large_files"))

# Concept DOI of the Zenodo record: it always resolves to the latest version, whose file links
# are read from the API at run time.
ZENODO_CONCEPT = "22757052"
API_URL = f"https://zenodo.org/api/records/{ZENODO_CONCEPT}"

# component -> (tarball, sha256, approximate size). Unpacks to large_files/<component>/.
MANIFEST = {
    "record": ("record.tar", "ad13b4702979fc7c44e3d3ce8cc145386631d5921192ab81ece131403040219d", "6.0 GB"),
    "feature_space_train": ("feature_space_train.tar", "f92e1a6b842005512139d2045eb65684264714d353d9eb8a46a75abc8e5aef81", "0.3 GB"),
    "feature_space_test": ("feature_space_test.tar", "155f702130799fbada807a873b6db2c76429a2914a98616c29deb5d811bfcf41", "2.0 GB"),
    "tac_activations": ("tac_activations.tar", "4e1d9c8fbe9835f4aa5ceff18373badc07a67eadf2feb7c7c5851532595d49f0", "1.9 GB"),
    "predictions_test_all_labels": ("predictions_test_all_labels.tar", "b7173b3c619663282d20feee4aa8bdc993f18405cc3fbaef4cb1e71e4cc5384e", "8 MB"),
    "tsne": ("tsne.tar", "31e20632dd65ed807e934c40050081d4877034495be62ecc943369ccdd3844ff", "9 MB"),
    "data": ("data.tar", "5ebf3e0073f1cfd7e01bba14daf4ccd5189fcbb5ace8a64fb3260a61951b84b3", "0.9 GB"),
    # retrained replicates used in the robustness analyses; unpack under large_files/replicates/
    "record_seeds": ("record_seeds.tar", "75ee16bc21a430943d9bb2d8e9871042862602215c3aed0b97bed9061ae1419d", "1.8 GB"),
    "record_targets": ("record_targets.tar", "bccd3a21389328023e190c0f553b803456e6386191ec593452d638057b594068", "0.8 GB"),
    "record_reruns": ("record_reruns.tar", "f4817555d03a76cbe9e350e682e23fc35f807990e526d39890df545de9edb410", "1.9 GB"),
    "vgg_records": ("vgg_records.tar", "6d688f9476a61a546b24ec0cb1c76ae693fcc93e9324e6fed0237071e98bdef3", "1.4 GB"),
}
REPLICATES = {"record_seeds", "record_targets", "record_reruns", "vgg_records"}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    last = [-1]

    def report(blocks, block_size, total):
        done = blocks * block_size
        step = done // (64 * 2**20)          # one line per 64 MiB keeps logs readable
        if step == last[0]:
            return
        last[0] = step
        if total > 0:
            sys.stdout.write(f"\r    {done / 2**30:6.2f} / {total / 2**30:6.2f} GiB")
        else:
            sys.stdout.write(f"\r    {done / 2**30:6.2f} GiB")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("components", nargs="*", default=list(MANIFEST),
                    help="components to fetch (default: all)")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    ap.add_argument("--no-replicates", action="store_true",
                    help="skip the retrained replicates, which only the robustness analyses read")
    ap.add_argument("--keep-tar", action="store_true", help="keep the tarballs after unpacking")
    args = ap.parse_args()
    if args.no_replicates:
        args.components = [c for c in args.components if c not in REPLICATES]

    if args.list:
        for name, (fname, digest, size) in MANIFEST.items():
            print(f"{name:28s} {fname:32s} {size:>7s}  sha256 {digest}")
        return

    unknown = [c for c in args.components if c not in MANIFEST]
    if unknown:
        raise SystemExit(f"unknown component(s): {unknown}; see --list")
    with urllib.request.urlopen(API_URL, timeout=120) as resp:
        record = json.load(resp)
    links = {f["key"]: f["links"]["self"] for f in record.get("files", [])}
    print(f"Zenodo record {record.get('id')} (version {record.get('metadata', {}).get('version', '?')})")

    tar_dir = LARGE_FILES / "tarballs"
    tar_dir.mkdir(parents=True, exist_ok=True)
    for name in args.components:
        fname, digest, size = MANIFEST[name]
        target_root = LARGE_FILES / "replicates" if name in REPLICATES else LARGE_FILES
        if (target_root / name).is_dir():
            print(f"[{name}] already unpacked at {target_root / name}, skipping")
            continue
        tar_path = tar_dir / fname
        if not (tar_path.exists() and sha256(tar_path) == digest):
            print(f"[{name}] downloading {fname} ({size}) ...")
            if fname not in links:
                raise SystemExit(f"[{name}] {fname} is not in the Zenodo record")
            download(links[fname], tar_path)
            actual = sha256(tar_path)
            if actual != digest:
                raise SystemExit(f"[{name}] checksum mismatch: expected {digest}, got {actual}")
        print(f"[{name}] checksum ok; unpacking to {target_root} ...")
        target_root.mkdir(parents=True, exist_ok=True)
        # unpack next to the destination and move the finished directory into place, so that an
        # interrupted run never leaves a partial directory that looks complete
        partial = target_root / f".{name}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            if hasattr(tarfile, "data_filter"):          # safe extraction filter where available
                tar.extractall(partial, filter="data")
            else:
                tar.extractall(partial)
        (partial / name).rename(target_root / name)
        shutil.rmtree(partial, ignore_errors=True)
        if not args.keep_tar:
            tar_path.unlink()
    print("done")


if __name__ == "__main__":
    main()
