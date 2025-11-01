from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use("Agg")

def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected metrics file at {path}. Run the notebooks to generate it first."
        )
    with path.open() as f:
        return json.load(f)


def build_dataframe(base: Path) -> pd.DataFrame:
    deeponet_metrics = _load_json(base / "deeponet_results.json")
    fno_metrics = _load_json(base / "fno_results.json")

    rows: List[Dict[str, float]] = [
        {
            "model": "DeepONet",
            "dataset": _pretty_dataset_name(deeponet_metrics["dataset"]),
            "train_rmse": float(deeponet_metrics["train_rmse"]),
            "test_rmse": float(deeponet_metrics["test_rmse"]),
        }
    ]

    for dataset, metrics in fno_metrics.items():
        rows.append(
            {
                "model": "FNO",
                "dataset": _pretty_dataset_name(dataset),
                "train_rmse": float(metrics["train_rmse"]),
                "test_rmse": float(metrics["test_rmse"]),
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "model"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _pretty_dataset_name(name: str) -> str:
    name = name.lower()
    if name == "grf":
        return "GRF"
    if name == "parametric":
        return "Parametric"
    return name.capitalize()


def plot_bars(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ["train_rmse", "test_rmse"]):
        pivot = df.pivot(index="dataset", columns="model", values=split)
        pivot = pivot.sort_index()
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{split.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent
    df = build_dataframe(base)
    print("Task 3 RMSE summary:")
    print(df.to_string(index=False))

    output_path = base / "task3_rmse_barplot.png"
    plot_bars(df, output_path)

if __name__ == "__main__":
    main()
