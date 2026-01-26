import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------- 1. Cost & Latency Profiles ----------

@dataclass
class CostProfile:
    tokens: int
    latency: float

class CostManager:
    """Tracks cumulative costs and latency for a simulation run."""
    
    # Profiles
    SYMBOLIC = CostProfile(tokens=0, latency=0.001)
    SMALL_MODEL = CostProfile(tokens=10, latency=0.05)
    LARGE_MODEL = CostProfile(tokens=100, latency=1.5)
    
    def __init__(self):
        self.total_tokens = 0
        self.total_latency = 0.0
        self.latency_breakdown = {
            "symbolic": 0.0,
            "small": 0.0,
            "large": 0.0
        }
        
    def reset(self):
        self.total_tokens = 0
        self.total_latency = 0.0
        self.latency_breakdown = {k: 0.0 for k in self.latency_breakdown}

    def log_symbolic(self, count=1):
        self.total_latency += self.SYMBOLIC.latency * count
        self.latency_breakdown["symbolic"] += self.SYMBOLIC.latency * count

    def log_small_model(self, count=1):
        self.total_tokens += self.SMALL_MODEL.tokens * count
        self.total_latency += self.SMALL_MODEL.latency * count
        self.latency_breakdown["small"] += self.SMALL_MODEL.latency * count

    def log_large_model(self, count=1):
        self.total_tokens += self.LARGE_MODEL.tokens * count
        self.total_latency += self.LARGE_MODEL.latency * count
        self.latency_breakdown["large"] += self.LARGE_MODEL.latency * count
        
    def get_stats(self):
        return {
            "tokens": self.total_tokens,
            "latency": self.total_latency,
            "latency_breakdown": self.latency_breakdown.copy()
        }

# ---------- 2. Data Simulation ----------

@dataclass
class QuerySample:
    id: int
    complexity: str # "Simple" or "Complex"
    ambiguity: float # 0.0 to 1.0 (Higher is harder)
    num_candidates: int = 20 # Number of potential paths
    
    # Hidden ground truth for simulation logic
    correct_path_index: int = 0

@dataclass
class DatasetProfile:
    name: str
    avg_ambiguity: float
    complexity_ratio: float # Ratio of "Complex" queries
    num_candidates_mean: int
    num_candidates_std: int

class DataGenerator:
    def __init__(self, profile: DatasetProfile, seed=42):
        self.profile = profile
        np.random.seed(seed)
        
    def generate_samples(self, n=100) -> List[QuerySample]:
        samples = []
        for i in range(n):
            # Complexity based on profile
            is_complex = np.random.rand() < self.profile.complexity_ratio
            complexity = "Complex" if is_complex else "Simple"
            
            # Ambiguity: Beta distribution tailored to profile mean
            # Simple heuristic: Normal distribution clipped [0, 1]
            ambiguity = np.random.normal(self.profile.avg_ambiguity, 0.15)
            ambiguity = np.clip(ambiguity, 0.05, 0.95)
            
            # Candidates count
            n_cand = int(np.random.normal(self.profile.num_candidates_mean, self.profile.num_candidates_std))
            n_cand = max(5, n_cand) # Minimum 5 candidates
            
            samples.append(QuerySample(
                id=i, 
                complexity=complexity, 
                ambiguity=ambiguity,
                num_candidates=n_cand 
            ))
        return samples

# Defined Profiles
PROFILES = {
    "MetaQA": DatasetProfile(
        name="MetaQA",
        avg_ambiguity=0.25,      # Clear entities, but multi-hop
        complexity_ratio=0.8,    # Mostly multi-hop
        num_candidates_mean=15,
        num_candidates_std=5
    ),
    "WebQSP": DatasetProfile(
        name="WebQSP",
        avg_ambiguity=0.65,      # Real web questions, ambiguous entities
        complexity_ratio=0.4,    # Mixed
        num_candidates_mean=30,  # Larger search space
        num_candidates_std=10
    ),
    "Logistics": DatasetProfile(
        name="Logistics",
        avg_ambiguity=0.45,       # IDs can be confusing, but structure is rigid
        complexity_ratio=0.9,     # Deep supply chains
        num_candidates_mean=50,   # Massive graph density
        num_candidates_std=15
    )
}
