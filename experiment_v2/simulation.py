
import time
import random
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from experiment_v2 import config

@dataclass
class APIUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    latency: float = 0.0
    cost: float = 0.0
    calls: int = 0

class CostMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CostMonitor, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.total_usage = APIUsage()
        self.session_usage = APIUsage() # For current run

    def record_usage(self, tokens_in, tokens_out, latency, model_type="large"):
        # Calculate Cost
        cost = 0.0
        if model_type == "large":
            cost = (tokens_in / 1000 * config.PRICE_LARGE_INPUT) + \
                   (tokens_out / 1000 * config.PRICE_LARGE_OUTPUT)
        elif model_type == "small":
            cost = ((tokens_in + tokens_out) / 1000 * config.PRICE_SMALL_MODEL)
        
        # Update Session
        self.session_usage.tokens_in += tokens_in
        self.session_usage.tokens_out += tokens_out
        self.session_usage.latency += latency
        self.session_usage.cost += cost
        self.session_usage.calls += 1

        # Update Total
        self.total_usage.tokens_in += tokens_in
        self.total_usage.tokens_out += tokens_out
        self.total_usage.latency += latency
        self.total_usage.cost += cost
        self.total_usage.calls += 1

    def get_session_stats(self):
        return {
            "tokens_total": self.session_usage.tokens_in + self.session_usage.tokens_out,
            "cost_usd": self.session_usage.cost,
            "latency_avg": self.session_usage.latency / max(1, self.session_usage.calls)
        }

class DataGenerator:
    """
    Generates semi-synthetic QA pairs that require reasoning.
    1-hop: Direct fact.
    Multi-hop: Requires bridging two concepts.
    """
    def __init__(self):
        self.templates_1hop = [
            ("What follows {A}?", "{B}", "{A} is followed by {B}"),
            ("Who created {A}?", "{B}", "{A} was created by {B}"),
            ("Where is {A} located?", "{B}", "{A} is located in {B}")
        ]
        self.entities = [
            ("Python", "Guido van Rossum"),
            ("Linux", "Linus Torvalds"),
            ("Tesla", "Elon Musk"),
            ("Facebook", "Mark Zuckerberg"),
            ("Microsoft", "Bill Gates"),
            ("Apple", "Steve Jobs"),
            ("Amazon", "Jeff Bezos"),
            ("Google", "Larry Page"),
            ("Netflix", "Reed Hastings"),
            ("SpaceX", "Elon Musk")
        ]
        
    def generate_batch(self, size=10) -> List[Dict[str, Any]]:
        batch = []
        for i in range(size):
            # Mix 1-hop and Multi-hop implied logic
            is_complex = random.random() > 0.5
            
            if not is_complex:
                # 1-Hop: "Who created Linux?"
                ent, ans = random.choice(self.entities)
                q = f"Who created {ent}?"
                batch.append({
                    "id": i,
                    "question": q,
                    "complexity": "1-hop",
                    "ground_truth": ans
                })
            else:
                # Multi-Hop: "What other company did the creator of Tesla start?"
                # Need to find Entity -> Creator -> Other Entity
                # Simplified for demo: Just ask a slightly harder question
                ent, creator = random.choice(self.entities)
                # Find other companies by same creator
                others = [e[0] for e in self.entities if e[1] == creator and e[0] != ent]
                if others:
                    target = others[0]
                    q = f"What other company did the founder of {ent} founded?"
                    batch.append({
                        "id": i,
                        "question": q,
                        "complexity": "multi-hop",
                        "ground_truth": target
                    })
                else:
                    # Fallback
                    q = f"Who is the CEO of {ent}?" # Assume creator matches roughly
                    batch.append({
                        "id": i,
                        "question": q,
                        "complexity": "1-hop",
                        "ground_truth": creator
                    })
        return batch
