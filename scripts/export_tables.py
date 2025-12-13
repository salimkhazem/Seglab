"""Aggregate multi-seed results into paper-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Results root directory.")
    parser.add_argument("--tables", default="tables", help="Output tables directory.")
    parser.add_argument(
        "--experiment",
        default="benchmark_matrix",
        help="Experiment name to export (default: benchmark_matrix). Use 'all' to export each experiment separately.",
    )
    args = parser.parse_args()

    results_root = Path(args.results)
    tables_dir = Path(args.tables)
    tables_dir.mkdir(exist_ok=True)

    df = collect_results(results_root)
    if df.empty:
        print("No results found. Run experiments first.")
        return

    # Keep the latest run per (experiment,dataset,model,seed)
    df = df.sort_values(["experiment", "dataset", "model", "seed", "run_id"])
    df = df.drop_duplicates(subset=["experiment", "dataset", "model", "seed"], keep="last")

    if args.experiment != "all":
        df = df[df["experiment"] == args.experiment].copy()
        if df.empty:
            print(f"No results found for experiment '{args.experiment}'.")
            return

    metrics = sorted([c for c in df.columns if c.startswith("test/")])
    extra = [c for c in ["trainable_params"] if c in df.columns]
    metric_cols = metrics + extra

    experiments = sorted(df["experiment"].unique()) if args.experiment == "all" else [args.experiment]
    for exp in experiments:
        df_exp = df[df["experiment"] == exp].copy()
        if df_exp.empty:
            continue
        agg = df_exp.groupby(["dataset", "model"])[metric_cols].agg(["mean", "std"])
        mean_std_csv = tables_dir / f"{exp}_mean_std.csv"
        tex_path = tables_dir / f"{exp}.tex"
        formatted_csv = tables_dir / f"{exp}_formatted.csv"
        formatted_tex = tables_dir / f"{exp}_formatted.tex"

        agg.to_csv(mean_std_csv)
        with open(tex_path, "w") as f:
            f.write(agg.to_latex(float_format="%.4f"))

        formatted = pd.DataFrame(index=agg.index)
        for col in metric_cols:
            m = agg[(col, "mean")]
            s = agg[(col, "std")]
            formatted[col] = [_format_mean_std(mv, sv) for mv, sv in zip(m.values, s.values)]
        formatted.to_csv(formatted_csv)
        with open(formatted_tex, "w") as f:
            f.write(formatted.to_latex(escape=False))

        # Backward-compatible aliases for the benchmark matrix (paper main table).
        if exp == "benchmark_matrix":
            (tables_dir / "benchmark_mean_std.csv").write_text(mean_std_csv.read_text())
            (tables_dir / "benchmark.tex").write_text(tex_path.read_text())
            (tables_dir / "benchmark_formatted.csv").write_text(formatted_csv.read_text())
            (tables_dir / "benchmark_formatted.tex").write_text(formatted_tex.read_text())

    print(f"Saved tables to {tables_dir}")


if __name__ == "__main__":
    main()
