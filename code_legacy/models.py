import numpy as np
from simulation import CostManager, QuerySample

class BaseReasoner:
    def __init__(self, name: str):
        self.name = name
        
    def solve(self, sample: QuerySample, cost_manager: CostManager) -> float:
        """
        Simulates solving the query.
        Returns: Accuracy (1.0 for correct, 0.0 for incorrect)
        """
        raise NotImplementedError

# ---------- Baseline 1: Standard ToG ----------

class StandardToG(BaseReasoner):
    def __init__(self):
        super().__init__("Standard ToG")
        self.beam_width = 3
        self.depth = 3
        
    def solve(self, sample: QuerySample, cost_manager: CostManager) -> float:
        # Logic: Beam search explores W * D paths using Large Model
        # Total calls = width * depth = 9 calls (simplified)
        
        num_calls = self.beam_width * self.depth
        cost_manager.log_large_model(num_calls)
        
        # Accuracy Simulation:
        # ToG is very robust. Even with high ambiguity, it performs well.
        # Simple: 100%, Complex: 95%
        base_acc = 1.0
        penalty = 0.05 * sample.ambiguity # Slight penalty for ambiguity
        return max(0.0, base_acc - penalty)

# ---------- Baseline 2: FrugalGPT ----------

class FrugalGPT(BaseReasoner):
    def __init__(self):
        super().__init__("FrugalGPT")
        
    def solve(self, sample: QuerySample, cost_manager: CostManager) -> float:
        # Logic: FrugalGPT as a Cascading Ranker
        # Stage 1: Score ALL candidates with Small Model
        candidates = sample.num_candidates
        cost_manager.log_small_model(candidates)
        
        # Simulate Confidence
        # If ambiguity is high, the "Gap" between best and second best is small (Low confidence)
        confidence = 1.0 - sample.ambiguity + np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0, 1)
        
        threshold = 0.6 # Lower threshold to be more aggressive in saving cost
        
        if confidence >= threshold:
            # Acceptance at Stage 1
            if sample.complexity == "Simple":
                return 0.95 
            else:
                return 0.65 # Small model risky for complex
        else:
            # Stage 2: Escalate
            # Usually, we don't re-score ALL 20. We rescore Top-K (e.g., 5).
            top_k = 5
            cost_manager.log_large_model(top_k)
            
            # High accuracy from Large Model
            return 1.0 - (0.02 * sample.ambiguity)

# ---------- Proposed: Tiered-Forest ----------

class TieredForest(BaseReasoner):
    def __init__(self):
        super().__init__("Tiered-Forest")
        
    def solve(self, sample: QuerySample, cost_manager: CostManager) -> float:
        candidates = sample.num_candidates # e.g., 20
        
        # --- Tier 1: Symbolic Filter ---
        # Logic: Fast rule-based filtering.
        # Cost Savings 1: Removes obviously wrong paths (e.g., wrong entity type)
        # Filters 50% of candidates.
        cost_manager.log_symbolic(candidates) # Latency for checking all
        remaining = int(candidates * 0.5) 
        
        # --- Tier 2: Semantic Scoring (Cross-Encoder) ---
        # Logic: Score remaining candidates with Small Model
        cost_manager.log_small_model(remaining) 
        
        # Simulate Scores generation
        # Correct path gets high score, others get low.
        # Ambiguity brings them closer.
        
        # Score distribution simulation
        correct_score = np.random.normal(0.9 - (0.2 * sample.ambiguity), 0.05)
        distractor_scores = np.random.normal(0.1 + (0.2 * sample.ambiguity), 0.1, remaining - 1)
        
        all_scores = np.concatenate(([correct_score], distractor_scores))
        
        # Dual-Threshold Logic
        # Discard < 0.2
        # Pass > 0.8
        # Ambiguity Zone: [0.2, 0.8]
        
        pass_threshold = 0.8
        discard_threshold = 0.2
        
        escalated_candidates = 0
        found_pass = False
        
        # Check scores
        for score in all_scores:
            if score > pass_threshold:
                found_pass = True
                 # If we found a high confidence "Pass", we usually stop or take it.
                 # For simulation, let's say we output it.
                break 
            elif score >= discard_threshold:
                # Ambiguity Zone -> Escalate
                escalated_candidates += 1
            # else: Discard (Cost Savings 2)
            
        # --- Tier 3: LLM Validation ---
        if found_pass:
            # Fast-Pass triggered. No LLM cost.
            # Accuracy check: Was the "Pass" actually the correct one?
            # If ambiguity is high, we might have false positive "Pass".
            # But mostly correct.
             return 0.98 if sample.complexity == "Simple" else 0.90
             
        elif escalated_candidates > 0:
            # Escalate only the ambiguous ones to Large Model
            # Cost Savings 3: Only calling LLM for a small subset, unlike ToG which calls for many/all
            cost_manager.log_large_model(escalated_candidates)
            
            # Logic: LLM sees filtered list, very high chance to pick correct one
            return 1.0 - (0.01 * sample.ambiguity)
            
        else:
            # All discarded? Fallback / Failure
            return 0.0
