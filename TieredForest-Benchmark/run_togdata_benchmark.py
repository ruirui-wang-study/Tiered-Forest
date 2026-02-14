#!/usr/bin/env python3
"""
Benchmark ToG vs Tiered-Forest on ToG datasets with a pluggable KG backend.

Default target backend is Wikidata.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src import config
from src.agents.forest_agent import TieredForestAgent
from src.agents.tog_agent import ToGAgent
from src.cost_monitor import CostMonitor
from src.data_loader_tog import ToGDatasetLoader
from src.graph_engine import MetaQAGraphEngine
from src.kg import create_kg_backend
from src.tog_eval import (
    build_eval_record,
    evaluate_prediction,
    evaluate_records,
    normalize_prediction,
)
from src.utils.cache_manager import LLMCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ToG-data benchmark for ToG vs Tiered-Forest"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="webqsp",
        choices=[
            "cwq",
            "webqsp",
            "grailqa",
            "simpleqa",
            "qald",
            "webquestions",
            "trex",
            "zeroshotre",
            "creak",
        ],
        help="ToG dataset key",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test", "all"],
        help="dataset split",
    )
    parser.add_argument("--limit", type=int, default=100, help="number of samples")
    parser.add_argument("--shuffle", action="store_true", help="shuffle samples")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    parser.add_argument(
        "--backend",
        type=str,
        default="wikidata",
        choices=["metaqa", "wikidata", "freebase"],
        help="KG backend type",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["tog", "tiered_forest"],
        choices=["tog", "tiered_forest"],
        help="agents to run",
    )
    parser.add_argument(
        "--tog_data_dir",
        type=str,
        default=os.path.join(config.ROOT_DIR, "ToG", "data"),
        help="path to ToG/data directory",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join("results", "togdata"),
        help="output directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="only load data and print run configuration",
    )

    # metaqa backend options
    parser.add_argument(
        "--metaqa_kb_path",
        type=str,
        default=os.path.join("data", "MetaQA", "kb.txt"),
        help="MetaQA kb path (used only when backend=metaqa)",
    )
    parser.add_argument(
        "--metaqa_cache_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="MetaQA graph cache dir (used only when backend=metaqa)",
    )

    # wikidata backend options
    parser.add_argument(
        "--wikidata_server_urls_file",
        type=str,
        default=config.WIKIDATA_SERVER_URLS_FILE,
        help="server urls file for wikidata backend",
    )

    # freebase backend options
    parser.add_argument(
        "--freebase_sparql_endpoint",
        type=str,
        default=config.FREEBASE_SPARQL_ENDPOINT,
        help="SPARQL endpoint for freebase backend",
    )

    # ToG parameters
    parser.add_argument("--tog_depth", type=int, default=2, help="ToG search depth")
    parser.add_argument("--tog_width", type=int, default=3, help="ToG search width")
    parser.add_argument("--tog_temperature", type=float, default=0.0, help="ToG LLM temperature")

    # Tiered-Forest parameters
    parser.add_argument("--t_drop", type=float, default=0.3, help="tiered-forest drop threshold")
    parser.add_argument("--t_pass", type=float, default=0.6, help="tiered-forest pass threshold")
    parser.add_argument(
        "--enable_small_model",
        action="store_true",
        help="enable Tier-2 small-model candidate generation (higher speed/cost tradeoff).",
    )

    return parser.parse_args()


def build_backend(args: argparse.Namespace):
    if args.backend == "metaqa":
        kb_path = (
            args.metaqa_kb_path
            if os.path.isabs(args.metaqa_kb_path)
            else os.path.join(os.path.dirname(__file__), args.metaqa_kb_path)
        )
        cache_dir = (
            args.metaqa_cache_dir
            if os.path.isabs(args.metaqa_cache_dir)
            else os.path.join(os.path.dirname(__file__), args.metaqa_cache_dir)
        )
        graph_engine = MetaQAGraphEngine(kb_path, cache_dir)
        return create_kg_backend("metaqa", graph_engine=graph_engine)

    if args.backend == "wikidata":
        return create_kg_backend(
            "wikidata",
            server_urls_file=args.wikidata_server_urls_file,
        )

    if args.backend == "freebase":
        return create_kg_backend(
            "freebase",
            sparql_endpoint=args.freebase_sparql_endpoint,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


def build_agents(args: argparse.Namespace, backend, cache_base_dir: str):
    agents = []

    if "tog" in args.agents:
        tog_cache = LLMCache(os.path.join(cache_base_dir, f"{args.dataset}_tog_cache.json"))
        agents.append(
            {
                "name": "ToG",
                "cache": tog_cache,
                "builder": lambda monitor: ToGAgent(
                    monitor=monitor,
                    graph_engine=None,
                    kg_backend=backend,
                    cache_manager=tog_cache,
                    depth=args.tog_depth,
                    width=args.tog_width,
                    temperature=args.tog_temperature,
                ),
            }
        )

    if "tiered_forest" in args.agents:
        forest_cache = LLMCache(
            os.path.join(cache_base_dir, f"{args.dataset}_tiered_forest_cache.json")
        )
        agents.append(
            {
                "name": "Tiered-Forest",
                "cache": forest_cache,
                "builder": lambda monitor: TieredForestAgent(
                    monitor=monitor,
                    graph_engine=None,
                    kg_backend=backend,
                    cache_manager=forest_cache,
                    t_drop=args.t_drop,
                    t_pass=args.t_pass,
                    enable_small_model=args.enable_small_model,
                ),
            }
        )

    return agents


def run() -> None:
    args = parse_args()

    print("=" * 90)
    print("ToG Data Benchmark (ToG vs Tiered-Forest)")
    print("=" * 90)
    print(f"Dataset: {args.dataset} | Split: {args.split} | Backend: {args.backend}")
    print(f"Limit: {args.limit} | Agents: {', '.join(args.agents)}")

    loader = ToGDatasetLoader(args.tog_data_dir)
    samples = loader.load(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print(f"Loaded samples: {len(samples)}")
    if not samples:
        raise RuntimeError("No samples loaded. Check dataset/split selection.")

    if args.dry_run:
        sample = samples[0]
        print("Dry run enabled. Sample preview:")
        print(f"  id={sample['id']}")
        print(f"  question={sample['question'][:120]}")
        print(f"  answers_count={len(sample['answers'])}")
        print(f"  topic_entity_count={len(sample['topic_entity'])}")
        print(f"  qid_topic_entity_count={len(sample['qid_topic_entity'])}")
        return

    backend = build_backend(args)

    # Prepare output dirs
    os.makedirs(args.result_dir, exist_ok=True)
    cache_base_dir = os.path.join("data", "cache", "togdata")
    os.makedirs(cache_base_dir, exist_ok=True)

    agents = build_agents(args, backend, cache_base_dir)
    if not agents:
        raise RuntimeError("No agents selected.")

    all_summary: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for agent_cfg in agents:
        name = agent_cfg["name"]
        print(f"\n{'-' * 90}")
        print(f"Running agent: {name}")
        print(f"{'-' * 90}")

        monitor = CostMonitor()
        agent = agent_cfg["builder"](monitor)
        detail_records: List[Dict[str, Any]] = []

        start = time.time()
        for sample in tqdm(samples, desc=f"{name}"):
            try:
                prediction = agent.solve(sample)
            except Exception as exc:
                prediction = f"ERROR: {exc}"

            answers = sample["answers"]
            correct = evaluate_prediction(prediction, answers)
            normalized_pred = normalize_prediction(prediction)
            detail = build_eval_record(
                sample_id=sample["id"],
                dataset=sample["dataset"],
                split=sample["split"],
                question=sample["question"],
                prediction=normalized_pred,
                answers=answers,
            )
            detail["agent"] = name
            detail["raw_prediction"] = prediction
            detail["correct"] = correct
            detail_records.append(detail)
            all_details.append(detail)

        elapsed = time.time() - start
        em_stats = evaluate_records(detail_records)
        usage = monitor.get_session_stats()

        summary = {
            "agent": name,
            "dataset": args.dataset,
            "split": args.split,
            "backend": args.backend,
            "samples": len(samples),
            "accuracy": em_stats["Exact Match"],
            "right": em_stats["Right Samples"],
            "error": em_stats["Error Samples"],
            "total_cost_usd": usage["cost_usd"],
            "avg_cost_usd": usage["cost_usd"] / max(1, len(samples)),
            "total_tokens": usage["tokens_total"],
            "avg_tokens": usage["tokens_total"] / max(1, len(samples)),
            "avg_latency_s": usage["latency_avg"],
            "total_time_s": elapsed,
            "calls_total": usage["calls_total"],
        }

        if name == "ToG":
            tog_stats = agent.get_stats()
            summary["avg_depth"] = tog_stats.get("avg_depth", 0.0)
            summary["llm_calls"] = tog_stats.get("llm_calls", 0)
            summary["avg_llm_calls"] = tog_stats.get("llm_calls", 0) / max(1, len(samples))
        elif name == "Tiered-Forest":
            tier_usage = agent.get_tier_usage()
            summary["tier1_pct"] = tier_usage.get("tier1", 0) / max(1, len(samples)) * 100
            summary["tier2_pct"] = tier_usage.get("tier2", 0) / max(1, len(samples)) * 100
            summary["tier3_pct"] = tier_usage.get("tier3", 0) / max(1, len(samples)) * 100

        all_summary.append(summary)
        agent_cfg["cache"].save()

        print(
            f"{name} => EM={summary['accuracy']:.2%}, "
            f"cost=${summary['total_cost_usd']:.6f}, "
            f"avg_latency={summary['avg_latency_s']:.2f}s"
        )

    summary_df = pd.DataFrame(all_summary)
    details_df = pd.DataFrame(all_details)

    summary_path = os.path.join(
        args.result_dir,
        f"summary_{args.dataset}_{args.backend}_{timestamp}.csv",
    )
    detailed_path = os.path.join(
        args.result_dir,
        f"detailed_{args.dataset}_{args.backend}_{timestamp}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(detailed_path, index=False)

    # Keep stable "latest" paths as well
    latest_summary = os.path.join(args.result_dir, "benchmark_summary.csv")
    latest_detailed = os.path.join(args.result_dir, "benchmark_detailed.csv")
    summary_df.to_csv(latest_summary, index=False)
    details_df.to_csv(latest_detailed, index=False)

    print("\n" + "=" * 90)
    print("Benchmark completed.")
    print(f"Summary:  {summary_path}")
    print(f"Detailed: {detailed_path}")
    print(f"Latest summary:  {latest_summary}")
    print(f"Latest detailed: {latest_detailed}")
    print("=" * 90)


if __name__ == "__main__":
    run()
