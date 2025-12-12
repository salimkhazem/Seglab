"""Aggregate multi-seed results into paper-ready tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def collect_results(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in results_root.glob("**/test_metrics.json"):
        try:
            rel = path.relative_to(results_root)
        except Exception:
            continue
        if len(rel.parts) < 6:
            continue
        experiment, dataset, model, seed_part = rel.parts[:4]
        run_id = "-".join(rel.parts[4:-1])
        if not seed_part.startswith("seed"):
            continue
        try:
            seed = int(seed_part.replace("seed", ""))
        except Exception:
            continue
        metrics = json.loads(path.read_text())
        row = {
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "run_id": run_id,
        }
        row.update(metrics)
        tp_path = path.parent / "trainable_params.txt"
        if tp_path.exists():
            try:
                row["trainable_params"] = int(tp_path.read_text().strip())
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    results_root = Path("results")
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    df = collect_results(results_root)
    if df.empty:
        print("No results found. Run experiments first.")
        return

    # Keep the latest run per (experiment,dataset,model,seed)
    df = df.sort_values(["experiment", "dataset", "model", "seed", "run_id"])
    df = df.drop_duplicates(subset=["experiment", "dataset", "model", "seed"], keep="last")

    metrics = [c for c in df.columns if c.startswith("test/")]
    extra = [c for c in ["trainable_params"] if c in df.columns]
    metric_cols = metrics + extra

    agg = df.groupby(["dataset", "model"])[metric_cols].agg(["mean", "std"])
    agg.to_csv(tables_dir / "benchmark_mean_std.csv")

    # Simple LaTeX export (multi-index columns)
    with open(tables_dir / "benchmark.tex", "w") as f:
        f.write(agg.to_latex(float_format="%.4f"))

    # Formatted mean±std table
    formatted = pd.DataFrame(index=agg.index)
    for col in metric_cols:
        m = agg[(col, "mean")]
        s = agg[(col, "std")]
        formatted[col] = [f"{mv:.4f}±{sv:.4f}" for mv, sv in zip(m.values, s.values)]
    formatted.to_csv(tables_dir / "benchmark_formatted.csv")
    with open(tables_dir / "benchmark_formatted.tex", "w") as f:
        f.write(formatted.to_latex(escape=False))
    print(f"Saved tables to {tables_dir}")


if __name__ == "__main__":
    main()
