"""Unpack only the files the benchmark needs from the downloaded archives.

    python -I src/lsb/extract.py

XES3G5M ships two multi-gigabyte "window sequence" files and several thousand images that
are not used; they are skipped.
"""
from __future__ import annotations

import os
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

XES_KEEP = {
    "XES3G5M/question_level/train_valid_sequences_quelevel.csv": "question_level__train_valid_sequences_quelevel.csv",
    "XES3G5M/question_level/test_quelevel.csv": "question_level__test_quelevel.csv",
    "XES3G5M/metadata/kc_routes_map.json": "metadata__kc_routes_map.json",
    "XES3G5M/metadata/questions.json": "metadata__questions.json",
}
DKT2_KEEP = {
    "dataset/assistments17/preprocessed_df.csv": "assistments17__preprocessed_df.csv",
    "dataset/ednet/preprocessed_df.csv": "ednet__preprocessed_df.csv",
}


def _write(dest: Path, data: bytes):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"  {dest.relative_to(RAW)}  {len(data)/1e6:.1f} MB", flush=True)


def xes3g5m():
    out = RAW / "xes3g5m" / "extracted"
    with tarfile.open(RAW / "xes3g5m" / "XES3G5M.tar.gz") as t:
        for m in t:
            if m.name in XES_KEEP and not (out / XES_KEEP[m.name]).exists():
                _write(out / XES_KEEP[m.name], t.extractfile(m).read())


def dkt2():
    out = RAW / "dkt2" / "extracted"
    with zipfile.ZipFile(RAW / "dkt2" / "dataset.zip") as z:
        for name, dest in DKT2_KEEP.items():
            if not (out / dest).exists():
                _write(out / dest, z.read(name))


def flat_zip(src: Path, out: Path):
    with zipfile.ZipFile(src) as z:
        for i in z.infolist():
            name = os.path.basename(i.filename)       # flatten: never trust paths inside an archive
            if name and not (out / name).exists():
                _write(out / name, z.read(i.filename))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("== xes3g5m"); xes3g5m()
    print("== dkt2"); dkt2()
    print("== oulad"); flat_zip(RAW / "oulad" / "oulad_uci.zip", RAW / "oulad" / "extracted")
    print("== slepemapy"); flat_zip(RAW / "slepemapy" / "2015-05-21.zip", RAW / "slepemapy" / "extracted")
