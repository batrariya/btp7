"""Reusable terminal/CSV/JSON reporting."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROWS = ["NDCG", "Recall", "Male NDCG", "Female NDCG", "Male Recall", "Female Recall", "JS-DP", "JS-EO"]


def format_table(model_name: str, epoch: int, metrics: dict, ks: list[int]) -> str:
    lines = ["=" * 60, f"[{model_name}] Epoch {epoch} - Test Set (after training)", "=" * 60,
             "Metric".ljust(24) + "".join(f"K={k}".rjust(12) for k in ks), "-" * 60]
    for row in ROWS:
        lines.append(row.ljust(24) + "".join(f"{metrics[row][k]:.6f}".rjust(12) for k in ks))
    return "\n".join(lines + ["=" * 60])


def save_results(directory: Path, metrics: dict, metadata: dict, ks: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "metrics": metrics}, handle, indent=2, default=str)
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *[f"K={k}" for k in ks]])
        writer.writeheader()
        for metric in ROWS:
            writer.writerow({"metric": metric, **{f"K={k}": metrics[metric][k] for k in ks}})
