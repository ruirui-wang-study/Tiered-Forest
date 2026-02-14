#!/usr/bin/env python3
"""
Analyze Tiered-Forest optimization results on ToG-data benchmark and generate charts.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ExperimentPaths:
    old_summary: Path
    old_detailed: Path
    new_summary: Path
    new_detailed: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ToG-data experiment and generate comparison charts."
    )
    parser.add_argument(
        "--old_summary",
        type=Path,
        default=Path("results/togdata/summary_webqsp_wikidata_20260213_083900.csv"),
    )
    parser.add_argument(
        "--old_detailed",
        type=Path,
        default=Path("results/togdata/detailed_webqsp_wikidata_20260213_083900.csv"),
    )
    parser.add_argument(
        "--new_summary",
        type=Path,
        default=Path("results/togdata/summary_webqsp_wikidata_20260214_104700.csv"),
    )
    parser.add_argument(
        "--new_detailed",
        type=Path,
        default=Path("results/togdata/detailed_webqsp_wikidata_20260214_104700.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/togdata/analysis_20260214"),
    )
    return parser.parse_args()


def ensure_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")


def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_detailed(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def by_agent(rows: List[Dict[str, str]], agent: str) -> Dict[str, Dict[str, str]]:
    return {row["id"]: row for row in rows if row.get("agent") == agent}


def question_prefix(question: str) -> str:
    q = str(question).strip().lower()
    if not q:
        return "other"
    return q.split()[0]


def build_metric_table(old_summary: pd.DataFrame, new_summary: pd.DataFrame) -> pd.DataFrame:
    old_tf = old_summary[old_summary["agent"] == "Tiered-Forest"].iloc[0]
    old_tog = old_summary[old_summary["agent"] == "ToG"].iloc[0]
    new_tf = new_summary[new_summary["agent"] == "Tiered-Forest"].iloc[0]

    rows = [
        {
            "model": "Tiered-Forest (Before)",
            "accuracy": float(old_tf["accuracy"]),
            "avg_cost_usd": float(old_tf["avg_cost_usd"]),
            "avg_latency_s": float(old_tf["avg_latency_s"]),
        },
        {
            "model": "ToG",
            "accuracy": float(old_tog["accuracy"]),
            "avg_cost_usd": float(old_tog["avg_cost_usd"]),
            "avg_latency_s": float(old_tog["avg_latency_s"]),
        },
        {
            "model": "Tiered-Forest (After)",
            "accuracy": float(new_tf["accuracy"]),
            "avg_cost_usd": float(new_tf["avg_cost_usd"]),
            "avg_latency_s": float(new_tf["avg_latency_s"]),
        },
    ]
    return pd.DataFrame(rows)


def build_prefix_accuracy(
    old_tf: Dict[str, Dict[str, str]],
    old_tog: Dict[str, Dict[str, str]],
    new_tf: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    all_ids = sorted(set(new_tf) & set(old_tf) & set(old_tog))
    by_prefix = defaultdict(lambda: {"n": 0, "old_tf": 0, "new_tf": 0, "tog": 0})

    for qid in all_ids:
        prefix = question_prefix(new_tf[qid]["question"])
        by_prefix[prefix]["n"] += 1
        by_prefix[prefix]["old_tf"] += int(normalize_bool(old_tf[qid]["correct"]))
        by_prefix[prefix]["new_tf"] += int(normalize_bool(new_tf[qid]["correct"]))
        by_prefix[prefix]["tog"] += int(normalize_bool(old_tog[qid]["correct"]))

    top_prefixes = sorted(by_prefix.items(), key=lambda x: x[1]["n"], reverse=True)[:8]
    records = []
    for prefix, stats in top_prefixes:
        n = stats["n"]
        records.append(
            {
                "prefix": prefix,
                "samples": n,
                "Tiered-Before": stats["old_tf"] / n,
                "Tiered-After": stats["new_tf"] / n,
                "ToG": stats["tog"] / n,
            }
        )
    return pd.DataFrame(records)


def build_transition_counts(
    old_tf: Dict[str, Dict[str, str]],
    new_tf: Dict[str, Dict[str, str]],
) -> Dict[str, int]:
    all_ids = sorted(set(new_tf) & set(old_tf))
    c = Counter()
    for qid in all_ids:
        old_ok = normalize_bool(old_tf[qid]["correct"])
        new_ok = normalize_bool(new_tf[qid]["correct"])
        if (not old_ok) and new_ok:
            c["Wrong -> Correct"] += 1
        elif old_ok and (not new_ok):
            c["Correct -> Wrong"] += 1
        elif old_ok and new_ok:
            c["Correct -> Correct"] += 1
        else:
            c["Wrong -> Wrong"] += 1
    return dict(c)


def build_head_to_head(
    new_tf: Dict[str, Dict[str, str]],
    old_tog: Dict[str, Dict[str, str]],
) -> Dict[str, int]:
    all_ids = sorted(set(new_tf) & set(old_tog))
    c = Counter()
    for qid in all_ids:
        new_ok = normalize_bool(new_tf[qid]["correct"])
        tog_ok = normalize_bool(old_tog[qid]["correct"])
        if new_ok and (not tog_ok):
            c["After TF wins"] += 1
        elif (not new_ok) and tog_ok:
            c["ToG wins"] += 1
        else:
            c["Tie"] += 1
    return dict(c)


def plot_main_metrics(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    models = df["model"].tolist()
    x = np.arange(len(models))
    colors = ["#8aa1b1", "#c46a45", "#4d8b59"]

    axes[0].bar(x, df["accuracy"].values * 100, color=colors)
    axes[0].set_title("Accuracy (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylim(0, 100)

    axes[1].bar(x, df["avg_cost_usd"].values, color=colors)
    axes[1].set_title("Avg Cost per Query (USD)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")

    axes[2].bar(x, df["avg_latency_s"].values, color=colors)
    axes[2].set_title("Avg Latency (s)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_prefix_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    prefixes = df["prefix"].tolist()
    x = np.arange(len(prefixes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, df["Tiered-Before"].values * 100, width=width, label="Tiered-Before")
    ax.bar(x, df["Tiered-After"].values * 100, width=width, label="Tiered-After")
    ax.bar(x + width, df["ToG"].values * 100, width=width, label="ToG")

    ax.set_title("Accuracy by Question Prefix")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\\n(n={n})" for p, n in zip(prefixes, df["samples"])], rotation=0)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_transition(counts: Dict[str, int], out_path: Path) -> None:
    order = ["Wrong -> Correct", "Correct -> Wrong", "Correct -> Correct", "Wrong -> Wrong"]
    values = [counts.get(k, 0) for k in order]
    colors = ["#4d8b59", "#c85252", "#7aa6c2", "#9a9a9a"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(order, values, color=colors)
    ax.set_title("Tiered-Forest Before vs After Transition")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_head_to_head(counts: Dict[str, int], out_path: Path) -> None:
    labels = ["After TF wins", "ToG wins", "Tie"]
    values = [counts.get(k, 0) for k in labels]
    colors = ["#4d8b59", "#c46a45", "#8d8d8d"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Head-to-Head: Tiered-Forest (After) vs ToG")
    ax.set_ylabel("Samples")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(
    out_path: Path,
    metrics: pd.DataFrame,
    transitions: Dict[str, int],
    head_to_head: Dict[str, int],
) -> None:
    before = metrics[metrics["model"] == "Tiered-Forest (Before)"].iloc[0]
    after = metrics[metrics["model"] == "Tiered-Forest (After)"].iloc[0]
    tog = metrics[metrics["model"] == "ToG"].iloc[0]

    acc_gain = (after["accuracy"] - before["accuracy"]) * 100
    vs_tog = (after["accuracy"] - tog["accuracy"]) * 100

    lines = [
        "# ToG-data Experiment Analysis",
        "",
        "## Core Findings",
        f"- Tiered-Forest accuracy improved from {before['accuracy']*100:.1f}% to {after['accuracy']*100:.1f}% (+{acc_gain:.1f} pp).",
        f"- Optimized Tiered-Forest is {vs_tog:+.1f} pp vs ToG ({tog['accuracy']*100:.1f}%).",
        f"- Avg latency dropped from {before['avg_latency_s']:.2f}s to {after['avg_latency_s']:.2f}s.",
        f"- Avg cost/query changed from ${before['avg_cost_usd']:.6f} to ${after['avg_cost_usd']:.6f}.",
        "",
        "## Transition Analysis (Before -> After)",
        f"- Wrong -> Correct: {transitions.get('Wrong -> Correct', 0)}",
        f"- Correct -> Wrong: {transitions.get('Correct -> Wrong', 0)}",
        f"- Correct -> Correct: {transitions.get('Correct -> Correct', 0)}",
        f"- Wrong -> Wrong: {transitions.get('Wrong -> Wrong', 0)}",
        "",
        "## Head-to-Head (After TF vs ToG)",
        f"- After TF wins: {head_to_head.get('After TF wins', 0)}",
        f"- ToG wins: {head_to_head.get('ToG wins', 0)}",
        f"- Tie: {head_to_head.get('Tie', 0)}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exp = ExperimentPaths(
        old_summary=args.old_summary,
        old_detailed=args.old_detailed,
        new_summary=args.new_summary,
        new_detailed=args.new_detailed,
        out_dir=args.out_dir,
    )
    ensure_paths([exp.old_summary, exp.old_detailed, exp.new_summary, exp.new_detailed])
    exp.out_dir.mkdir(parents=True, exist_ok=True)

    old_summary = load_summary(exp.old_summary)
    new_summary = load_summary(exp.new_summary)
    old_rows = load_detailed(exp.old_detailed)
    new_rows = load_detailed(exp.new_detailed)

    old_tf = by_agent(old_rows, "Tiered-Forest")
    old_tog = by_agent(old_rows, "ToG")
    new_tf = by_agent(new_rows, "Tiered-Forest")

    metric_df = build_metric_table(old_summary, new_summary)
    prefix_df = build_prefix_accuracy(old_tf, old_tog, new_tf)
    transition_counts = build_transition_counts(old_tf, new_tf)
    head_to_head_counts = build_head_to_head(new_tf, old_tog)

    plot_main_metrics(metric_df, exp.out_dir / "01_main_metrics.png")
    plot_prefix_accuracy(prefix_df, exp.out_dir / "02_accuracy_by_prefix.png")
    plot_transition(transition_counts, exp.out_dir / "03_before_after_transition.png")
    plot_head_to_head(head_to_head_counts, exp.out_dir / "04_head_to_head_vs_tog.png")

    write_report(
        exp.out_dir / "analysis_summary.md",
        metric_df,
        transition_counts,
        head_to_head_counts,
    )
    print(f"Saved analysis to: {exp.out_dir}")


if __name__ == "__main__":
    main()

