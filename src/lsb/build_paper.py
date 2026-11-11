"""Assemble manuscript/paper.md from manuscript/paper_template.md and the results.

Every number in the manuscript is a placeholder resolved from results/summary.json, and
every results table is included from results/tables, so the text cannot drift away from
the analysis output.

    {{a/b/c}}            value at summary["a"]["b"]["c"]
    {{a/b|.3f}}          the same with a Python format spec ( |+.3f  |.1%  |d )
    {{table:name}}       contents of results/tables/name.md
    {{file:rel/path}}    contents of a text file relative to the project root

    python -I src/lsb/build_paper.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("LSB_RESULTS", ROOT / "results"))
MAN = ROOT / "manuscript"


def lookup(summary, path):
    cur = summary
    for part in path.split("/"):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = cur[int(part)]
    return cur


def main():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    text = (MAN / "paper_template.md").read_text(encoding="utf-8")
    missing = []

    def repl(m):
        body = m.group(1).strip()
        try:
            if body.startswith("table:"):
                return (RES / "tables" / f"{body[6:]}.md").read_text(encoding="utf-8")
            if body.startswith("file:"):
                return (ROOT / body[5:]).read_text(encoding="utf-8").rstrip()
            path, _, fmt = body.partition("|")
            val = lookup(summary, path.strip())
            if fmt:
                return format(val, fmt.strip()).replace("-", "−") if isinstance(val, (int, float)) else str(val)
            return str(val)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{body}  ({type(e).__name__}: {e})")
            return f"[[MISSING {body}]]"

    out = text
    for _ in range(4):                       # included section files contain placeholders too
        new = re.sub(r"\{\{(.+?)\}\}", repl, out)
        if new == out:
            break
        out = new
    (MAN / "paper.md").write_text(out, encoding="utf-8")
    words = len(re.findall(r"\b\w+\b", re.sub(r"\|.*\|", " ", out)))
    print(f"wrote manuscript/paper.md  (~{words} words outside tables)")
    if missing:
        print("UNRESOLVED PLACEHOLDERS:")
        for m in missing:
            print("  ", m)
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
