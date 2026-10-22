"""
Download and unpack Caltech-101 into backend/data/categories/.

The dataset is ~140 MB and is deliberately not committed, so this script is how
a fresh clone gets playable.

    uv run python scripts/fetch_data.py [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "backend" / "data" / "categories" / "101_ObjectCategories"


def report(block_num: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block_num * block_size, total)
    pct = 100 * done / total
    sys.stdout.write(f"\r  {pct:5.1f}%  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if the data is present")
    args = ap.parse_args()

    if TARGET.is_dir() and any(TARGET.iterdir()) and not args.force:
        n = sum(1 for _ in TARGET.rglob("*.jpg"))
        print(f"Already present: {TARGET} ({n} images). Use --force to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / "caltech-101.zip"

        print(f"Downloading Caltech-101 from {URL}")
        urllib.request.urlretrieve(URL, archive, reporthook=report)
        print("\n  extracting ...")

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmpdir)

        # The official zip nests the categories inside a tarball.
        tarballs = list(tmpdir.rglob("101_ObjectCategories.tar.gz"))
        if tarballs:
            with tarfile.open(tarballs[0]) as tf:
                tf.extractall(tarballs[0].parent, filter="data")

        found = [p for p in tmpdir.rglob("101_ObjectCategories") if p.is_dir()]
        if not found:
            sys.exit("Could not find 101_ObjectCategories in the archive -- layout changed?")

        TARGET.parent.mkdir(parents=True, exist_ok=True)
        if TARGET.exists():
            shutil.rmtree(TARGET)
        shutil.move(str(found[0]), str(TARGET))

    n = sum(1 for _ in TARGET.rglob("*.jpg"))
    print(f"Done: {n} images in {TARGET}")


if __name__ == "__main__":
    main()
