"""
Ablation Study Model Variants

Implements all 8 ablation configurations:
- E0: Full System (baseline)
- A1: No-Tier1 (remove symbolic layer)
- A2: No-Tier2 (remove semantic layer)
- A3: No-Dynamic (static thresholds)
- A4: Two-Tier-Only (no Tier 1, same as A1)
- A5: Tier1+LLM (skip Tier 2, same as A2)
- A6: Single-Threshold (traditional cascade)
- A7: Aggressive-Pass (low tau_high for cost savings)
"""

import time
from typing import Optional, Dict, Any
from experiment_v2.models import call_llm, call_small_model, SymbolicLayer
from experiment_v2.component_library import create_tier2_component
from experiment_v2.routing_strategies import create_routing_strategy
from experiment_v2.metrics_tracker import MetricsTracker


class BaseRoutingModel:
    """
    Base class for all routing models in ablation study
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsTracker()
        self.name = config.get('name', 'UnnamedModel')
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        """
        Unified solve interface
        
        Args:
            question: Input question
            ground_truth: Ground truth answer (for recall tracking)
            **kwargs: Additional context (load_factor, budget_factor, etc.)
            
        Returns:
            Answer string
        """
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        return self.metrics.export()
    
    def reset_metrics(self):
        """Reset metrics for new evaluation"""
        self.metrics.reset()
    
    def get_name(self) -> str:
        """Get model name"""
        return self.name


# ===== E0: Full System =====

class TieredForestFull(BaseRoutingModel):
    """
    E0: Complete Tiered-Forest system
    - Tier 1: Symbolic layer
    - Tier 2: Cross-Encoder semantic scoring
    - Routing: Dual-threshold dynamic
    """
    def __init__(self):
        super().__init__({
            'name': 'E0_Full',
            'tier1_enabled': True,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'dual_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('cross_encoder')
        self.router = create_routing_strategy('dual_dynamic', 
                                              tau_low=0.2, tau_high=0.7, dynamic=True)
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        tier1_answer = None
        
        # Tier 1: Symbolic
        self.metrics.start_timer('tier1')
        self.metrics.record_tier_call('tier1')
        sym_ans = self.symbolic.check(question)
        self.metrics.stop_timer('tier1')
        
        if sym_ans:
            self.metrics.record_route('tier1_hit')
            tier1_answer = sym_ans
            self.metrics.record_tier1_decision(True, sym_ans, sym_ans, ground_truth)
            self.metrics.stop_timer('total')
            return sym_ans
        
        # Tier 1.5: Candidate Generation
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2: Semantic Scoring
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Routing Decision
        context = kwargs.get('context', {})
        decision = self.router.decide(score, context)
        self.metrics.record_route(decision)
        self.metrics.record_thresholds(self.router.get_thresholds())
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        elif decision == "discard":
            # Optionally return empty or fallback to LLM
            pass
        
        # Tier 3: LLM
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== A1: No-Tier1 =====

class TieredForestNoTier1(BaseRoutingModel):
    """
    A1: Remove Tier 1 symbolic layer
    Directly start with Tier 2 semantic scoring
    """
    def __init__(self):
        super().__init__({
            'name': 'A1_NoTier1',
            'tier1_enabled': False,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'dual_dynamic'
        })
        self.tier2 = create_tier2_component('cross_encoder')
        self.router = create_routing_strategy('dual_dynamic', 
                                              tau_low=0.2, tau_high=0.7, dynamic=True)
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        self.metrics.start_timer('total')
        
        # Skip Tier 1, go directly to candidate generation
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2: Semantic Scoring
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
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3: LLM
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        self.metrics.stop_timer('total')
        
        return answer


# ===== A2: No-Tier2 =====

class TieredForestNoTier2(BaseRoutingModel):
    """
    A2: Remove Tier 2 semantic layer
    Tier 1 -> directly to Tier 3 LLM
    """
    def __init__(self):
        super().__init__({
            'name': 'A2_NoTier2',
            'tier1_enabled': True,
            'tier2_component': None,
            'routing_strategy': None
        })
        self.symbolic = SymbolicLayer()
    
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
        
        # Skip Tier 2, go directly to Tier 3
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        self.metrics.record_route('tier3_direct')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== A3: No-Dynamic =====

class TieredForestNoDynamic(BaseRoutingModel):
    """
    A3: Full system but with static thresholds (no dynamic adjustment)
    """
    def __init__(self):
        super().__init__({
            'name': 'A3_NoDynamic',
            'tier1_enabled': True,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'dual_static'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('cross_encoder')
        self.router = create_routing_strategy('dual_dynamic', 
                                              tau_low=0.2, tau_high=0.7, dynamic=False)
    
    def solve(self, question: str, ground_truth: str = "", **kwargs) -> str:
        # Same as E0 but router.dynamic = False
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
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Routing (static thresholds)
        decision = self.router.decide(score, {})
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== A4: Two-Tier-Only (alias for A1) =====

class TieredForestTwoTier(TieredForestNoTier1):
    """
    A4: Two-tier only (equivalent to A1: No-Tier1)
    Demonstrates that three-tier is better than two-tier
    """
    def __init__(self):
        super().__init__()
        self.name = 'A4_TwoTier'
        self.config['name'] = 'A4_TwoTier'


# ===== A5: Tier1+LLM (alias for A2) =====

class TieredForestTier1LLM(TieredForestNoTier2):
    """
    A5: Tier 1 + LLM only (equivalent to A2: No-Tier2)
    Tests if middle layer is necessary
    """
    def __init__(self):
        super().__init__()
        self.name = 'A5_Tier1LLM'
        self.config['name'] = 'A5_Tier1LLM'


# ===== A6: Single-Threshold =====

class TieredForestSingleThreshold(BaseRoutingModel):
    """
    A6: Full system but with single threshold (traditional cascade)
    """
    def __init__(self):
        super().__init__({
            'name': 'A6_SingleThreshold',
            'tier1_enabled': True,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'single'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('cross_encoder')
        self.router = create_routing_strategy('single', tau=0.5)
    
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
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Single threshold routing
        decision = self.router.decide(score, {})
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== A7: Aggressive-Pass =====

class TieredForestAggressive(BaseRoutingModel):
    """
    A7: Aggressive Fast-Pass strategy (lower tau_high)
    Maximizes cost savings at the expense of accuracy
    """
    def __init__(self):
        super().__init__({
            'name': 'A7_Aggressive',
            'tier1_enabled': True,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'aggressive'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = create_tier2_component('cross_encoder')
        self.router = create_routing_strategy('aggressive', tau_low=0.2, tau_high=0.5)
    
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
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2
        self.metrics.start_timer('tier2')
        self.metrics.record_tier_call('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Aggressive routing (low tau_high = 0.5)
        decision = self.router.decide(score, {})
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.record_tier1_decision(False, None, candidate, ground_truth)
            self.metrics.stop_timer('total')
            return candidate
        
        # Tier 3
        self.metrics.start_timer('tier3')
        self.metrics.record_tier_call('tier3')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        
        self.metrics.record_tier1_decision(False, None, answer, ground_truth)
        self.metrics.stop_timer('total')
        return answer


# ===== Factory Function =====

def create_ablation_model(variant: str) -> BaseRoutingModel:
    """
    Factory function to create ablation model variants
    
    Args:
        variant: One of 'E0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'
        
    Returns:
        BaseRoutingModel instance
    """
    models = {
        'E0': TieredForestFull,
        'A1': TieredForestNoTier1,
        'A2': TieredForestNoTier2,
        'A3': TieredForestNoDynamic,
        'A4': TieredForestTwoTier,
        'A5': TieredForestTier1LLM,
        'A6': TieredForestSingleThreshold,
        'A7': TieredForestAggressive
    }
    
    if variant not in models:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(models.keys())}")
    
    return models[variant]()
