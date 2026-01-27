"""
Enhanced Metrics Tracker for Ablation Study

Provides detailed tracking of:
- Latency breakdown by tier
- Routing decision distribution
- Token consumption by tier
- Tier 1 recall/precision metrics
- Threshold adjustment history
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict


class MetricsTracker:
    """
    Comprehensive metrics tracking for ablation experiments
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        # Timing metrics
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        
        # Routing metrics
        self.routes: List[str] = []
        self.scores: Dict[str, List[float]] = defaultdict(list)
        
        # Token metrics (by tier)
        self.tier_tokens = {'tier1': 0, 'tier2': 0, 'tier3': 0, 'candidate_gen': 0}
        self.tier_calls = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        
        # Tier 1 recall tracking
        self.tier1_decisions: List[Dict[str, Any]] = []
        
        # Threshold history (for dynamic routing)
        self.threshold_history: List[Dict[str, float]] = []
    
    # ===== Timer Methods =====
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.active_timers[name] = time.time()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stop a named timer and record elapsed time
        
        Returns:
            Elapsed time in seconds, or None if timer wasn't started
        """
        if name in self.active_timers:
            elapsed = time.time() - self.active_timers[name]
            self.timers[name].append(elapsed)
            del self.active_timers[name]
            return elapsed
        return None
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named timer"""
        if name not in self.timers or not self.timers[name]:
            return {'mean': 0.0, 'total': 0.0, 'count': 0}
        
        times = self.timers[name]
        return {
            'mean': sum(times) / len(times),
            'total': sum(times),
            'count': len(times),
            'min': min(times),
            'max': max(times)
        }
    
    # ===== Routing Methods =====
    
    def record_route(self, decision: str):
        """Record a routing decision"""
        self.routes.append(decision)
    
    def record_score(self, name: str, value: float):
        """Record a score (e.g., tier2_score)"""
        self.scores[name].append(value)
    
    def get_routing_distribution(self) -> Dict[str, float]:
        """Get distribution of routing decisions"""
        if not self.routes:
            return {}
        
        total = len(self.routes)
        distribution = {}
        for route in set(self.routes):
            distribution[route] = self.routes.count(route) / total
        
        return distribution
    
    # ===== Token Tracking =====
    
    def record_tokens(self, tier: str, tokens: int):
        """Record token consumption for a tier"""
        if tier in self.tier_tokens:
            self.tier_tokens[tier] += tokens
    
    def record_tier_call(self, tier: str):
        """Record a call to a specific tier"""
        if tier in self.tier_calls:
            self.tier_calls[tier] += 1
    
    # ===== Tier 1 Recall Tracking =====
    
    def record_tier1_decision(self, tier1_hit: bool, tier1_answer: Optional[str],
                             final_answer: str, ground_truth: str):
        """
        Record Tier 1 decision for recall analysis
        
        Args:
            tier1_hit: Whether Tier 1 rule was triggered
            tier1_answer: Answer from Tier 1 (if hit)
            final_answer: Final answer from the system
            ground_truth: Ground truth answer
        """
        self.tier1_decisions.append({
            'tier1_hit': tier1_hit,
            'tier1_answer': tier1_answer,
            'final_answer': final_answer,
            'ground_truth': ground_truth
        })
    
    def compute_tier1_recall(self) -> Dict[str, float]:
        """
        Compute Tier 1 recall, precision, and false negative rate
        
        Returns:
            Dict with recall, precision, false_negative_rate
        """
        if not self.tier1_decisions:
            return {'recall': 0.0, 'precision': 0.0, 'false_negative_rate': 0.0}
        
        total_correct = 0
        tier1_preserved_correct = 0
        tier1_discarded_correct = 0
        tier1_hits = 0
        tier1_correct_hits = 0
        
        for record in self.tier1_decisions:
            is_final_correct = record['ground_truth'].lower() in record['final_answer'].lower()
            tier1_hit = record['tier1_hit']
            
            if is_final_correct:
                total_correct += 1
                if tier1_hit:
                    tier1_preserved_correct += 1
                else:
                    tier1_discarded_correct += 1
            
            if tier1_hit:
                tier1_hits += 1
                if record['tier1_answer'] and record['ground_truth'].lower() in record['tier1_answer'].lower():
                    tier1_correct_hits += 1
        
        recall = tier1_preserved_correct / total_correct if total_correct > 0 else 0.0
        precision = tier1_correct_hits / tier1_hits if tier1_hits > 0 else 0.0
        false_negative_rate = tier1_discarded_correct / total_correct if total_correct > 0 else 0.0
        
        return {
            'recall': recall,
            'precision': precision,
            'false_negative_rate': false_negative_rate,
            'tier1_hit_rate': tier1_hits / len(self.tier1_decisions)
        }
    
    # ===== Threshold Tracking =====
    
    def record_thresholds(self, thresholds: Dict[str, float]):
        """Record current threshold values (for dynamic routing)"""
        self.threshold_history.append({
            'timestamp': time.time(),
            **thresholds
        })
    
    # ===== Export Methods =====
    
    def export(self) -> Dict[str, Any]:
        """
        Export all metrics as a dictionary
        
        Returns:
            Comprehensive metrics dictionary
        """
        return {
            # Latency metrics
            'latency_breakdown': {
                name: self.get_timer_stats(name)
                for name in self.timers.keys()
            },
            'latency_total': self.get_timer_stats('total').get('mean', 0.0),
            
            # Routing metrics
            'routing_distribution': self.get_routing_distribution(),
            'average_scores': {
                name: sum(scores) / len(scores) if scores else 0.0
                for name, scores in self.scores.items()
            },
            
            # Token metrics
            'tier_tokens': self.tier_tokens.copy(),
            'tier_calls': self.tier_calls.copy(),
            'total_tokens': sum(self.tier_tokens.values()),
            
            # Tier 1 recall
            'tier1_recall_metrics': self.compute_tier1_recall(),
            
            # Threshold history
            'threshold_adjustments': len(self.threshold_history),
            'final_thresholds': self.threshold_history[-1] if self.threshold_history else {}
        }
    
    def export_summary(self) -> str:
        """
        Export a human-readable summary
        
        Returns:
            Formatted string summary
        """
        metrics = self.export()
        
        summary = []
        summary.append("=== Metrics Summary ===")
        summary.append(f"Total Latency: {metrics['latency_total']:.2f}s")
        summary.append(f"Total Tokens: {metrics['total_tokens']}")
        summary.append(f"\nRouting Distribution:")
        for route, pct in metrics['routing_distribution'].items():
            summary.append(f"  {route}: {pct:.1%}")
        
        tier1_metrics = metrics['tier1_recall_metrics']
        summary.append(f"\nTier 1 Performance:")
        summary.append(f"  Recall: {tier1_metrics['recall']:.1%}")
        summary.append(f"  Precision: {tier1_metrics['precision']:.1%}")
        summary.append(f"  Hit Rate: {tier1_metrics['tier1_hit_rate']:.1%}")
        
        return "\n".join(summary)


class RecallTracker:
    """
    Standalone Tier 1 recall tracker for detailed analysis
    (Can be used independently or integrated with MetricsTracker)
    """
    def __init__(self):
        self.decisions: List[Dict[str, Any]] = []
    
    def record(self, tier1_hit: bool, tier1_answer: Optional[str],
               final_answer: str, ground_truth: str, question: str = ""):
        """Record a decision with optional question for debugging"""
        self.decisions.append({
            'question': question,
            'tier1_hit': tier1_hit,
            'tier1_answer': tier1_answer,
            'final_answer': final_answer,
            'ground_truth': ground_truth
        })
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform detailed recall analysis
        
        Returns:
            Detailed analysis including examples of false negatives
        """
        if not self.decisions:
            return {}
        
        false_negatives = []
        false_positives = []
        true_positives = []
        
        for record in self.decisions:
            gt = record['ground_truth'].lower()
            final = record['final_answer'].lower()
            tier1_ans = (record['tier1_answer'] or "").lower()
            
            is_final_correct = gt in final
            tier1_hit = record['tier1_hit']
            is_tier1_correct = gt in tier1_ans if tier1_ans else False
            
            if is_final_correct and not tier1_hit:
                false_negatives.append(record)
            elif tier1_hit and not is_tier1_correct:
                false_positives.append(record)
            elif tier1_hit and is_tier1_correct:
                true_positives.append(record)
        
        return {
            'total_samples': len(self.decisions),
            'false_negatives': len(false_negatives),
            'false_positives': len(false_positives),
            'true_positives': len(true_positives),
            'false_negative_examples': false_negatives[:5],  # First 5 examples
            'false_positive_examples': false_positives[:5]
        }
