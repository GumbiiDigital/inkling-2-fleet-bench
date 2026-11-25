#!/usr/bin/env python3
"""Export normalized CSV tables from data/benchmarks.json."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TABLES = [
    "speed_ladder",
    "precision_ladder",
    "concurrency",
    "speculative_decoding",
    "needle_in_haystack",
    "evals_and_vision",
    "soak_and_power",
    "power_ladder_rtx6000",
    "varloop",
]


def main() -> None:
    payload = json.loads((DATA_DIR / "benchmarks.json").read_text(encoding="utf-8"))
    for name in TABLES:
        rows = payload[name]
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        out_path = DATA_DIR / f"{name}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
