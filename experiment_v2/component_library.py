"""
Component Library for Tiered-Forest Ablation Study

Provides pluggable Tier 2 components for fair comparison:
- CrossEncoderComponent: Semantic similarity via Cross-Encoder
- SBERTComponent: Sentence embeddings via SBERT
- JaccardComponent: Token overlap (zero-parameter baseline)
- LLaMaSelfConfidenceComponent: LLM self-evaluation
"""

import re
import numpy as np
from typing import Protocol


class Tier2Component(Protocol):
    """
    Protocol for Tier 2 scoring components
    """
    def score(self, question: str, candidate: str) -> float:
        """
        Score the relevance/confidence of a candidate answer
        
        Args:
            question: Input question
            candidate: Candidate answer to score
            
        Returns:
            float: Confidence score in [0, 1]
        """
        ...
    
    def get_name(self) -> str:
        """Return component name for logging"""
        ...


class CrossEncoderComponent:
    """
    Cross-Encoder based semantic scorer (default Tier 2)
    """
    def __init__(self, model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        try:
            from sentence_transformers import CrossEncoder
            self.encoder = CrossEncoder(model_name)
            self.available = True
        except Exception as e:
            print(f"Warning: CrossEncoder initialization failed: {e}")
            self.encoder = None
            self.available = False
    
    def score(self, question: str, candidate: str) -> float:
        if not self.available or not self.encoder:
            return 0.5  # Fallback to neutral score
        
        try:
            pred = self.encoder.predict([(question, candidate)])
            # Handle both scalar and array outputs
            if isinstance(pred, (list, np.ndarray)):
                pred = pred[0] if len(pred) > 0 else 0.0
            # Convert logit to probability via sigmoid
            score = 1 / (1 + np.exp(-float(pred)))
            return float(score)
        except Exception as e:
            print(f"CrossEncoder scoring error: {e}")
            return 0.5
    
    def get_name(self) -> str:
        return "CrossEncoder-TinyBERT"


class SBERTComponent:
    """
    SBERT (Sentence-BERT) based semantic scorer
    Uses cosine similarity between question and candidate embeddings
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except Exception as e:
            print(f"Warning: SBERT initialization failed: {e}")
            self.model = None
            self.available = False
    
    def score(self, question: str, candidate: str) -> float:
        if not self.available or not self.model:
            return 0.5
        
        try:
            embeddings = self.model.encode([question, candidate])
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            # Normalize from [-1, 1] to [0, 1]
            score = (similarity + 1) / 2
            return float(score)
        except Exception as e:
            print(f"SBERT scoring error: {e}")
            return 0.5
    
    def get_name(self) -> str:
        return "SBERT-MiniLM"


class JaccardComponent:
    """
    Jaccard similarity based scorer (zero-parameter baseline)
    Computes token overlap between question and candidate
    """
    def score(self, question: str, candidate: str) -> float:
        try:
            q_tokens = set(question.lower().split())
            c_tokens = set(candidate.lower().split())
            
            if not q_tokens or not c_tokens:
                return 0.0
            
            intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            
            score = intersection / union if union > 0 else 0.0
            return float(score)
        except Exception as e:
            print(f"Jaccard scoring error: {e}")
            return 0.0
    
    def get_name(self) -> str:
        return "Jaccard"


class LLaMaSelfConfidenceComponent:
    """
    LLaMA-based self-confidence scorer
    Uses the small model to generate answer and self-evaluate confidence
    """
    def __init__(self, call_small_model_fn):
        """
        Args:
            call_small_model_fn: Function to call small model (e.g., from models.py)
        """
        self.call_small_model = call_small_model_fn
    
    def score(self, question: str, candidate: str) -> float:
        """
        Ask LLaMA to rate its confidence in the candidate answer
        """
        try:
            prompt = (
                f"Question: {question}\n"
                f"Answer: {candidate}\n"
                f"Rate your confidence in this answer (0.0 to 1.0):"
            )
            response = self.call_small_model(prompt, max_tokens=10)
            
            # Extract confidence score from response
            match = re.search(r'0\.\d+|1\.0|0|1', response)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            else:
                return 0.5  # Default to neutral if parsing fails
        except Exception as e:
            print(f"LLaMA self-confidence error: {e}")
            return 0.5
    
    def get_name(self) -> str:
        return "LLaMA-SelfConfidence"


# Factory function for easy component creation
def create_tier2_component(component_type: str, **kwargs):
    """
    Factory function to create Tier 2 components
    
    Args:
        component_type: One of 'cross_encoder', 'sbert', 'jaccard', 'llama'
        **kwargs: Additional arguments for component initialization
        
    Returns:
        Tier2Component instance
    """
    components = {
        'cross_encoder': CrossEncoderComponent,
        'sbert': SBERTComponent,
        'jaccard': JaccardComponent,
        'llama': LLaMaSelfConfidenceComponent
    }
    
    if component_type not in components:
        raise ValueError(f"Unknown component type: {component_type}. "
                        f"Available: {list(components.keys())}")
    
    return components[component_type](**kwargs)
