"""
Fair Baseline Models for Component Comparison

Implements baseline configurations for fair comparison:
- B2: FrugalGPT-Fair (two-tier with Cross-Encoder)
- B3: FrugalGPT-SBERT (two-tier with SBERT)
- B5: Tiered-Forest-LLaMA (three-tier with LLaMA self-confidence)
- B6: Tiered-Forest-SBERT (three-tier with SBERT)
- B7: Tiered-Forest-Jaccard (three-tier with Jaccard)

These baselines enable:
1. Architecture comparison (two-tier vs three-tier)
2. Component comparison (Cross-Encoder vs SBERT vs Jaccard vs LLaMA)
3. Separation of architecture contribution from component contribution
"""

import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from experiment_v2.models import call_llm, call_small_model, SymbolicLayer
from experiment_v2.component_library import create_tier2_component
from experiment_v2.routing_strategies import create_routing_strategy
# Import BaseRoutingModel from new location
from experiments.phase1_ablation.code.ablation_models import BaseRoutingModel


# ===== B2: FrugalGPT-Fair =====

class FrugalGPTFair(BaseRoutingModel):
    """
    B2: Fair version of FrugalGPT using Cross-Encoder
    
    Architecture: Two-tier cascade
    - Stage 1: Cross-Encoder scoring
    - Stage 2: DeepSeek LLM
    
    This is the "fair" comparison to Tiered-Forest because it uses
    the same Tier 2 component (Cross-Encoder) but lacks Tier 1.
    """
    def __init__(self, threshold: float = 0.6):
        super().__init__({
            'name': 'B2_FrugalGPT_Fair',
            'architecture': 'two_tier',
            'tier2_component': 'cross_encoder',
            'threshold': threshold
        })
        self.tier2 = create_tier2_component('cross_encoder')
        self.threshold = threshold
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        
        # Stage 1: Generate candidate and score with Cross-Encoder
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Question: {question}\nAnswer directly with the entity/value only. No sentences.", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Single threshold decision
        if score > self.threshold:
            self.metrics.record_route('fast_pass')
            self.metrics.stop_timer('total')
            return candidate
        
        # Stage 2: LLM
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        self.metrics.record_route('escalate')
        answer = call_llm(f"Question: {question}\nAnswer directly and concisely. If listing entities, use commas. Do not explain.")
        self.metrics.stop_timer('tier3')
        self.metrics.stop_timer('total')
        
        return answer


# ===== B3: FrugalGPT-SBERT =====

class FrugalGPTSBERT(BaseRoutingModel):
    """
    B3: FrugalGPT with SBERT instead of Cross-Encoder
    
    Tests component swap in two-tier architecture
    """
    def __init__(self, threshold: float = 0.6):
        super().__init__({
            'name': 'B3_FrugalGPT_SBERT',
            'architecture': 'two_tier',
            'tier2_component': 'sbert',
            'threshold': threshold
        })
        self.tier2 = create_tier2_component('sbert')
        self.threshold = threshold
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        
        # Stage 1: SBERT scoring
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Question: {question}\nAnswer directly with the entity/value only. No sentences.", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        if score > self.threshold:
            self.metrics.record_route('fast_pass')
            self.metrics.stop_timer('total')
            return candidate
        
        # Stage 2: LLM
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        self.metrics.record_route('escalate')
        answer = call_llm(f"Question: {question}\nAnswer directly and concisely. If listing entities, use commas. Do not explain.")
        self.metrics.stop_timer('tier3')
        self.metrics.stop_timer('total')
        
        return answer


# ===== B5: Tiered-Forest-LLaMA =====

class TieredForestLLaMA(BaseRoutingModel):
    """
    B5: Tiered-Forest with LLaMA self-confidence instead of Cross-Encoder
    
    Architecture: Three-tier
    - Tier 1: Symbolic
    - Tier 2: LLaMA self-confidence scoring
    - Tier 3: DeepSeek LLM
    
    This tests whether the architecture advantage holds even with
    a different (potentially weaker) Tier 2 component.
    """
    def __init__(self):
        super().__init__({
            'name': 'B5_TieredForest_LLaMA',
            'tier1_enabled': True,
            'tier2_component': 'llama',
            'routing_strategy': 'dual_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('llama', call_small_model_fn=call_small_model)
        self.router = create_routing_strategy('dual_dynamic', 
                                              tau_low=0.2, tau_high=0.7, dynamic=True)
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        
        # Tier 1: Symbolic
        self.metrics.start_timer('tier1')
        self.metrics.record_tier_call('tier1')
        sym_ans = self.symbolic.check(question)
        self.metrics.stop_timer('tier1')
        
        if sym_ans:
            self.metrics.record_route('tier1_hit')
            self.metrics.record_tier1_decision(True, sym_ans, sym_ans, ground_truth)
            self.metrics.stop_timer('total')
            return sym_ans
        
        # Tier 1.5: Candidate generation
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Question: {question}\nAnswer directly with the entity/value only. No sentences.", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2: LLaMA self-confidence scoring
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Routing
        context = kwargs.get('context', {})
        decision = self.router.decide(score, context)
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3: LLM
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Question: {question}\nAnswer directly and concisely. If listing entities, use commas. Do not explain.")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== B6: Tiered-Forest-SBERT =====

class TieredForestSBERT(BaseRoutingModel):
    """
    B6: Tiered-Forest with SBERT instead of Cross-Encoder
    
    Tests component swap in three-tier architecture
    """
    def __init__(self):
        super().__init__({
            'name': 'B6_TieredForest_SBERT',
            'tier1_enabled': True,
            'tier2_component': 'sbert',
            'routing_strategy': 'dual_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('sbert')
        self.router = create_routing_strategy('dual_dynamic', 
                                              tau_low=0.2, tau_high=0.7, dynamic=True)
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        
        # Tier 1
        self.metrics.start_timer('tier1')
        self.metrics.record_tier_call('tier1')
        sym_ans = self.symbolic.check(question)
        self.metrics.stop_timer('tier1')
        
        if sym_ans:
            self.metrics.record_route('tier1_hit')
            self.metrics.record_tier1_decision(True, sym_ans, sym_ans, ground_truth)
            self.metrics.stop_timer('total')
            return sym_ans
        
        # Candidate generation
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Question: {question}\nAnswer directly with the entity/value only. No sentences.", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2: SBERT
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Routing
        context = kwargs.get('context', {})
        decision = self.router.decide(score, context)
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Question: {question}\nAnswer directly and concisely. If listing entities, use commas. Do not explain.")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== B7: Tiered-Forest-Jaccard =====

class TieredForestJaccard(BaseRoutingModel):
    """
    B7: Tiered-Forest with Jaccard similarity (zero-parameter baseline)
    
    This is the most lightweight configuration, testing whether
    the architecture can work with minimal Tier 2 complexity.
    """
    def __init__(self):
        super().__init__({
            'name': 'B7_TieredForest_Jaccard',
            'tier1_enabled': True,
            'tier2_component': 'jaccard',
            'routing_strategy': 'dual_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('jaccard')
        self.router = create_routing_strategy('dual_dynamic', 
                                              tau_low=0.2, tau_high=0.7, dynamic=True)
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        
        # Tier 1
        self.metrics.start_timer('tier1')
        self.metrics.record_tier_call('tier1')
        sym_ans = self.symbolic.check(question)
        self.metrics.stop_timer('tier1')
        
        if sym_ans:
            self.metrics.record_route('tier1_hit')
            self.metrics.record_tier1_decision(True, sym_ans, sym_ans, ground_truth)
            self.metrics.stop_timer('total')
            return sym_ans
        
        # Candidate generation
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Question: {question}\nAnswer directly with the entity/value only. No sentences.", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2: Jaccard
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Routing
        context = kwargs.get('context', {})
        decision = self.router.decide(score, context)
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Question: {question}\nAnswer directly and concisely. If listing entities, use commas. Do not explain.")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== Factory Function =====

def create_baseline_model(variant: str) -> BaseRoutingModel:
    """
    Factory function to create baseline model variants
    
    Args:
        variant: One of 'B2', 'B3', 'B5', 'B6', 'B7'
        
    Returns:
        BaseRoutingModel instance
    """
    models = {
        'B2': FrugalGPTFair,
        'B3': FrugalGPTSBERT,
        'B5': TieredForestLLaMA,
        'B6': TieredForestSBERT,
        'B7': TieredForestJaccard
    }
    
    if variant not in models:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(models.keys())}")
    
    return models[variant]()
