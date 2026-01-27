"""
Routing Strategies for Tiered-Forest Ablation Study

Provides different routing decision strategies:
- DualThresholdDynamicRouting: Full system with fuzzy zone
- SingleThresholdRouting: Traditional cascade (A6)
- AggressiveRouting: Low threshold for cost optimization (A7)
- NoRouting: Direct escalation (for A2, A5)
"""

from typing import Literal, Dict, Any


RoutingDecision = Literal["fast_pass", "escalate", "discard"]


class RoutingStrategy:
    """
    Base class for routing strategies
    """
    def decide(self, score: float, context: Dict[str, Any] = None) -> RoutingDecision:
        """
        Make routing decision based on score and context
        
        Args:
            score: Tier 2 confidence score [0, 1]
            context: Optional context (load_factor, budget_factor, etc.)
            
        Returns:
            RoutingDecision: "fast_pass", "escalate", or "discard"
        """
        raise NotImplementedError
    
    def get_thresholds(self) -> Dict[str, float]:
        """Return current threshold values for logging"""
        return {}


class DualThresholdDynamicRouting(RoutingStrategy):
    """
    Full Tiered-Forest routing strategy with:
    - Dual thresholds (tau_low, tau_high) creating a fuzzy zone
    - Optional dynamic threshold adjustment based on load/budget
    """
    def __init__(self, tau_low: float = 0.2, tau_high: float = 0.7, 
                 dynamic: bool = True, lambda1: float = 0.1, lambda2: float = 0.15):
        """
        Args:
            tau_low: Lower threshold (discard below this)
            tau_high: Upper threshold (fast-pass above this)
            dynamic: Enable dynamic threshold adjustment
            lambda1: Load response coefficient
            lambda2: Budget control coefficient
        """
        self.tau_low_base = tau_low
        self.tau_high_base = tau_high
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.dynamic = dynamic
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.adjustment_count = 0
    
    def decide(self, score: float, context: Dict[str, Any] = None) -> RoutingDecision:
        context = context or {}
        
        # Dynamic threshold adjustment
        if self.dynamic:
            self._adjust_thresholds(context)
        
        # Routing logic
        if score > self.tau_high:
            return "fast_pass"
        elif score < self.tau_low:
            return "discard"
        else:
            return "escalate"
    
    def _adjust_thresholds(self, context: Dict[str, Any]):
        """
        Dynamically adjust thresholds based on system state
        
        Context keys:
            - load_factor: Current load / baseline load (1.0 = normal)
            - budget_factor: Remaining budget / initial budget (1.0 = normal)
        """
        load_factor = context.get('load_factor', 1.0)
        budget_factor = context.get('budget_factor', 1.0)
        
        # Increase tau_high under high load (more fast-pass to reduce latency)
        # Decrease tau_high under budget pressure (more LLM calls for quality)
        self.tau_high = self.tau_high_base + self.lambda1 * (load_factor - 1.0)
        self.tau_high = max(0.5, min(0.9, self.tau_high))  # Clamp to [0.5, 0.9]
        
        # Decrease tau_low under budget pressure (discard more aggressively)
        self.tau_low = self.tau_low_base - self.lambda2 * (1.0 - budget_factor)
        self.tau_low = max(0.1, min(0.3, self.tau_low))  # Clamp to [0.1, 0.3]
        
        # Ensure fuzzy zone exists
        if self.tau_high <= self.tau_low + 0.1:
            self.tau_high = self.tau_low + 0.2
        
        self.adjustment_count += 1
    
    def get_thresholds(self) -> Dict[str, float]:
        return {
            'tau_low': self.tau_low,
            'tau_high': self.tau_high,
            'tau_low_base': self.tau_low_base,
            'tau_high_base': self.tau_high_base,
            'adjustment_count': self.adjustment_count
        }


class SingleThresholdRouting(RoutingStrategy):
    """
    Single threshold routing (A6: Single-Threshold variant)
    Traditional cascade strategy: pass if score > tau, else escalate
    """
    def __init__(self, tau: float = 0.5):
        """
        Args:
            tau: Single threshold value
        """
        self.tau = tau
    
    def decide(self, score: float, context: Dict[str, Any] = None) -> RoutingDecision:
        return "fast_pass" if score > self.tau else "escalate"
    
    def get_thresholds(self) -> Dict[str, float]:
        return {'tau': self.tau}


class AggressiveRouting(RoutingStrategy):
    """
    Aggressive Fast-Pass routing (A7: Aggressive-Pass variant)
    Lower tau_high to maximize cost savings at the expense of accuracy
    """
    def __init__(self, tau_low: float = 0.2, tau_high: float = 0.5):
        """
        Args:
            tau_low: Lower threshold (discard)
            tau_high: Upper threshold (fast-pass) - LOWER than default
        """
        self.tau_low = tau_low
        self.tau_high = tau_high
    
    def decide(self, score: float, context: Dict[str, Any] = None) -> RoutingDecision:
        if score > self.tau_high:
            return "fast_pass"
        elif score < self.tau_low:
            return "discard"
        else:
            return "escalate"
    
    def get_thresholds(self) -> Dict[str, float]:
        return {'tau_low': self.tau_low, 'tau_high': self.tau_high}


class NoRouting(RoutingStrategy):
    """
    No routing strategy - always escalate to Tier 3
    Used for A2 (No-Tier2) and A5 (Tier1+LLM) variants
    """
    def decide(self, score: float, context: Dict[str, Any] = None) -> RoutingDecision:
        return "escalate"
    
    def get_thresholds(self) -> Dict[str, float]:
        return {}


# Factory function
def create_routing_strategy(strategy_type: str, **kwargs) -> RoutingStrategy:
    """
    Factory function to create routing strategies
    
    Args:
        strategy_type: One of 'dual_dynamic', 'single', 'aggressive', 'none'
        **kwargs: Strategy-specific parameters
        
    Returns:
        RoutingStrategy instance
    """
    strategies = {
        'dual_dynamic': DualThresholdDynamicRouting,
        'single': SingleThresholdRouting,
        'aggressive': AggressiveRouting,
        'none': NoRouting
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}. "
                        f"Available: {list(strategies.keys())}")
    
    return strategies[strategy_type](**kwargs)
