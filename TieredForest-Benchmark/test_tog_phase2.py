#!/usr/bin/env python3
"""
Phase-2 smoke checks:
1) KG backend abstraction wiring
2) Agent input compatibility (string vs ToG-style sample dict)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.cost_monitor import CostMonitor
from src.graph_engine import MetaQAGraphEngine
from src.kg import MetaQABackend, create_kg_backend
from src.utils.cache_manager import LLMCache
from src.agents.forest_agent import TieredForestAgent
from src.agents.tog_agent import ToGAgent


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(base_dir, "data", "MetaQA", "kb.txt")
    cache_dir = os.path.join(base_dir, "data", "processed")
    llm_cache_path = os.path.join(base_dir, "data", "cache", "phase2_test_cache.json")

    graph_engine = MetaQAGraphEngine(kb_path, cache_dir)
    metaqa_backend = MetaQABackend(graph_engine)
    factory_backend = create_kg_backend("metaqa", graph_engine=graph_engine)

    assert metaqa_backend.find_entity("once were warriors"), "metaqa backend should resolve entities"
    assert factory_backend.name == "metaqa", "factory should build metaqa backend"

    monitor = CostMonitor()
    cache = LLMCache(llm_cache_path)

    forest = TieredForestAgent(
        monitor=monitor,
        graph_engine=None,
        cache_manager=cache,
        kg_backend=metaqa_backend,
        t_drop=0.3,
        t_pass=0.6,
    )

    sample = {
        "question": "who directed this movie?",
        "topic_entity": {"m.test": "once were warriors"},
        "qid_topic_entity": {},
    }
    answer = forest.solve(sample)
    assert isinstance(answer, str) and len(answer.strip()) > 0, "forest agent should accept sample input"

    tog = ToGAgent(
        monitor=monitor,
        graph_engine=None,
        cache_manager=cache,
        kg_backend=metaqa_backend,
        depth=1,
        width=2,
    )
    question, seeds = tog._normalize_query_input(sample)
    assert question == sample["question"]
    assert "once were warriors" in [s.lower() for s in seeds]

    print("Phase-2 checks passed.")


if __name__ == "__main__":
    main()
