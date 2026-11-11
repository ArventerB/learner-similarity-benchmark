"""Download the public datasets used in the benchmark.

Every source goes into its own directory under data/raw/ and is recorded in
data/raw/MANIFEST.json (URL, size in bytes, SHA-256, retrieval date).

Sources and licences (check the licence pages before redistributing):
  oulad    Open University Learning Analytics Dataset, CC BY 4.0
           https://analyse.kmi.open.ac.uk/open_dataset
  ptadisc  PTADisc (NeurIPS 2023 D&B), CC BY-NC-SA 4.0
           https://huggingface.co/datasets/whning0513/PTADisc
  xes3g5m  XES3G5M (NeurIPS 2023 D&B), MIT
           https://github.com/ai4ed/XES3G5M
  dkt2     Pre-processed Assist17 / EdNet sample / Comp released with DKT2 (code MIT;
           EdNet is CC BY-NC 4.0, Comp is PTADisc CC BY-NC-SA 4.0)
           https://github.com/zyy-2001/DKT2

Usage:  python -I src/lsb/download.py [oulad ptadisc xes3g5m dkt2]
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "learner-similarity-benchmark/0.1 (research)"}

HF = "https://huggingface.co/datasets/whning0513/PTADisc/resolve/main/"
PTA_BASE = "task specific dataset/non_programming_dataset/"
PTA_CROSS = {
    "cpp": "cross_course_datasets/baiteng_C++程序设计-92",
    "c": "cross_course_datasets/baiteng_C程序设计-63",
    "java": "cross_course_datasets/baiteng_Java程序设计-172",
    "python": "cross_course_datasets/baiteng_Python程序设计-924247594312409088",
    "ds": "cross_course_datasets/baiteng_数据结构与算法-1",
}
PTA_CD = {
    "comp_thinking": "baseline_dataset_for_CD/Computational_thinking",
    "database": "baseline_dataset_for_CD/Database_technology_and_application",
    "linux": "baseline_dataset_for_CD/Linux_system",
    "probability": "baseline_dataset_for_CD/Probability_and_statistic",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name: str, url: str, path: Path) -> None:
    man_path = RAW / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    man[str(path.relative_to(RAW)).replace("\\", "/")] = {
        "source": name,
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "retrieved": _dt.date.today().isoformat(),
    }
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")


def fetch(name: str, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(RAW)}", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
    tmp.replace(dest)
    record(name, url, dest)
    print(f"[ok]   {dest.relative_to(RAW)}  {n/1e6:.1f} MB", flush=True)


def get_oulad() -> None:
    # The link on the Open University page returned HTTP 404 on 2026-11-11, so the
    # UCI Machine Learning Repository copy (dataset 349, CC BY 4.0) is used instead.
    fetch("oulad",
          "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
          RAW / "oulad" / "oulad_uci.zip")


def get_ptadisc() -> None:
    for fn in ["LICENSE", "README.md"]:
        fetch("ptadisc", HF + fn, RAW / "ptadisc" / fn)
    for key, sub in PTA_CROSS.items():
        for fn in ["train.csv", "valid.csv", "test.csv", "item.csv", "info.json"]:
            url = HF + urllib.parse.quote(PTA_BASE + sub + "/" + fn)
            fetch("ptadisc", url, RAW / "ptadisc" / "cross_course" / key / fn)
    for key, sub in PTA_CD.items():
        for fn in ["dataset.csv", "item.csv", "info.json"]:
            url = HF + urllib.parse.quote(PTA_BASE + sub + "/" + fn)
            fetch("ptadisc", url, RAW / "ptadisc" / "single_course" / key / fn)


def get_gdrive(name: str, file_id: str, dest: Path) -> None:
    import gdown  # imported lazily: only needed for Google Drive sources

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(RAW)}", flush=True)
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=True)
    record(name, url, dest)
    print(f"[ok]   {dest.relative_to(RAW)}  {dest.stat().st_size/1e6:.1f} MB", flush=True)


def get_xes3g5m() -> None:
    get_gdrive("xes3g5m", "1eFiIYyh5O2V90RA0brammGH6EpHvPDQe", RAW / "xes3g5m" / "XES3G5M.tar.gz")


def get_dkt2() -> None:
    get_gdrive("dkt2", "1PMikGhRwSVAFc0319vxGkoZslM_jvYI_", RAW / "dkt2" / "dataset.zip")


def get_assist09() -> None:
    # ASSISTments 2009-2010 skill-builder data ("corrected, collapsed" release), linked from
    # https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
    get_gdrive("assist09", "1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE",
               RAW / "assist09" / "skill_builder_data_corrected_collapsed.csv")


def get_assist15() -> None:
    # ASSISTments 2015 skill-builder data, linked from
    # https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data
    get_gdrive("assist15", "0B_hO8cnpcIMgUGZzRnh3bHJrSjQ",
               RAW / "assist15" / "2015_100_skill_builders_main_problems.csv")


def get_slepemapy() -> None:
    # Adaptive geography practice data set (Papousek, Pelanek & Stanislav, JLA 2016), linked from
    # https://www.fi.muni.cz/adaptivelearning/?a=data
    fetch("slepemapy", "http://www.fi.muni.cz/adaptivelearning/data/slepemapy/2015-05-21.zip",
          RAW / "slepemapy" / "2015-05-21.zip")


SOURCES = {"oulad": get_oulad, "ptadisc": get_ptadisc, "xes3g5m": get_xes3g5m, "dkt2": get_dkt2,
           "assist09": get_assist09, "assist15": get_assist15, "slepemapy": get_slepemapy}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    wanted = sys.argv[1:] or list(SOURCES)
    for w in wanted:
        print(f"== {w}", flush=True)
        try:
            SOURCES[w]()
        except Exception as e:  # keep going: one failed source must not block the others
            print(f"[FAIL] {w}: {type(e).__name__}: {e}", flush=True)
