"""Metrics Module: Advanced semantic similarity and distributional analysis.

Implements:
1. WassersteinAnalyzer: 2-Wasserstein distance for comparing LLM vs expert embeddings
2. ReasoningEvaluator: Extracts and validates reasoning steps (SymCoT-specific)
3. FactualityEvaluator: NLI-based fact verification
4. SemanticAlignmentMetrics: Composable metrics for semantic analysis
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

try:
    import pot  # Python Optimal Transport
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    logger.warning("POT (Python Optimal Transport) not available. Wasserstein distance disabled.")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("SentenceTransformer not available. Embedding computation disabled.")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class WassersteinResult:
    """Result of Wasserstein distance computation."""
    distance_2d: float  # Full 2-Wasserstein distance
    distance_1d_pca: Optional[float] = None  # PCA-reduced (for interpretability)
    cost_matrix: Optional[np.ndarray] = None  # Cost matrix used
    transport_plan: Optional[np.ndarray] = None  # Optimal transport plan
    samples_llm: int = 0  # Number of LLM samples
    samples_expert: int = 0  # Number of expert samples


class BaseMetricEvaluator(ABC):
    """Abstract base for metric evaluators."""
    
    @abstractmethod
    def evaluate(self, llm_output: str, expert_reference: str) -> float:
        """Compute metric between LLM output and expert reference."""
        pass


class WassersteinAnalyzer(BaseMetricEvaluator):
    """
    Computes 2-Wasserstein distance between LLM and expert answers.
    
    Mathematical Foundation:
    Let μ and ν be empirical distributions of LLM and expert embeddings respectively.
    The 2-Wasserstein distance is defined as:
    
    W_2(μ, ν) = [min_{π ∈ Π(μ,ν)} E[||X - Y||^2]]^(1/2)
    
    where Π(μ,ν) is the set of all couplings with marginals μ and ν.
    
    We use the Earth Mover's Distance (EMD) algorithm via POT library for exact computation.
    Lower values indicate better semantic alignment between LLM output and expert standard.
    
    Reference:
    Peyré, G., Cuturi, M. (2019): "Computational Optimal Transport"
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        pca_components: Optional[int] = 307,
        cost_metric: str = "euclidean",
    ):
        """
        Initialize Wasserstein analyzer.
        
        Args:
            embedding_model: SentenceTransformer model name
            pca_components: PCA projection dimension (None = no PCA)
            cost_metric: Distance metric for cost matrix ("euclidean", "cosine")
        """
        self.embedding_model = embedding_model
        self.pca_components = pca_components
        self.cost_metric = cost_metric
        
        if ST_AVAILABLE:
            self._model = SentenceTransformer(embedding_model)
            logger.info(f"WassersteinAnalyzer initialized with {embedding_model}")
        else:
            logger.error("SentenceTransformer required for WassersteinAnalyzer")
    
    def _embed_sentences(self, text: str) -> np.ndarray:
        """
        Embed text into sentence vectors.
        
        Args:
            text: Input text (single sentence or multi-sentence)
        
        Returns:
            Array of shape (n_sentences, embedding_dim)
        """
        sentences = self._split_sentences(text)
        embeddings = self._model.encode(sentences, convert_to_numpy=True)
        return embeddings
    
    @staticmethod
    def _split_sentences(text: str, max_len: int = 512) -> List[str]:
        """Simple sentence splitting (can be enhanced with NLTK)."""
        # Naive split on periods, exclamation, question marks
        import re
        sentences = re.split(r'[.!?]+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def evaluate(self, llm_output: str, expert_reference: str) -> float:
        """
        Evaluate semantic alignment via 2-Wasserstein distance.
        
        Args:
            llm_output: LLM-generated answer
            expert_reference: Expert gold-standard answer
        
        Returns:
            Wasserstein distance (lower is better)
        """
        if not ST_AVAILABLE:
            logger.warning("Embedding model unavailable. Returning dummy metric.")
            return 0.0
        
        try:
            llm_embeddings = self._embed_sentences(llm_output)
            expert_embeddings = self._embed_sentences(expert_reference)
            result = self._compute_wasserstein(llm_embeddings, expert_embeddings)
            return result.distance_2d
        except Exception as e:
            logger.error(f"Wasserstein computation failed: {e}")
            return float('inf')
    
    def _compute_wasserstein(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> WassersteinResult:
        """
        Compute 2-Wasserstein distance between two point sets.
        
        Args:
            X: LLM embeddings (n, d)
            Y: Expert embeddings (m, d)
        
        Returns:
            WassersteinResult with distance and transport plan
        """
        if not POT_AVAILABLE:
            logger.warning("POT library not available. Returning dummy result.")
            return WassersteinResult(distance_2d=0.0, samples_llm=len(X), samples_expert=len(Y))
        
        # Compute cost matrix
        if self.cost_metric == "euclidean":
            cost_matrix = self._euclidean_cost_matrix(X, Y)
        elif self.cost_metric == "cosine":
            cost_matrix = self._cosine_cost_matrix(X, Y)
        else:
            cost_matrix = self._euclidean_cost_matrix(X, Y)
        
        try:
            # Compute optimal transport
            transport_plan = pot.emd(
                np.ones(len(X)) / len(X),
                np.ones(len(Y)) / len(Y),
                cost_matrix,
            )
            
            # Wasserstein distance = sqrt(trace(cost * transport_plan))
            distance = np.sqrt(np.sum(cost_matrix * transport_plan))
            
            return WassersteinResult(
                distance_2d=distance,
                cost_matrix=cost_matrix,
                transport_plan=transport_plan,
                samples_llm=len(X),
                samples_expert=len(Y),
            )
        except Exception as e:
            logger.error(f"EMD computation failed: {e}")
            return WassersteinResult(
                distance_2d=float('inf'),
                samples_llm=len(X),
                samples_expert=len(Y),
            )
    
    @staticmethod
    def _euclidean_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Euclidean distance cost matrix."""
        # ||x - y||^2
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True).T
        cost = X_norm + Y_norm - 2 * X @ Y.T
        return np.sqrt(np.maximum(cost, 0))  # Numerical stability
    
    @staticmethod
    def _cosine_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Cosine distance cost matrix."""
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        similarity = X_norm @ Y_norm.T
        return 1 - np.clip(similarity, -1, 1)  # 1 - cosine_similarity


class ReasoningEvaluator(BaseMetricEvaluator):
    """
    Extracts and validates reasoning chains, particularly for SymCoT.
    
    For SymCoT responses, validates:
    1. Number of valid reasoning steps
    2. Logical consistency of derived facts
    3. Verification quality of FOL-based checks
    """
    
    def __init__(self, max_reasoning_depth: int = 10):
        self.max_reasoning_depth = max_reasoning_depth
    
    def evaluate(self, llm_output: str, expert_reference: str = "") -> float:
        """
        Compute reasoning quality as percentage of valid reasoning steps.
        
        Args:
            llm_output: LLM response (may contain structured reasoning)
            expert_reference: Not used, included for interface compatibility
        
        Returns:
            Percentage of valid reasoning steps (0-100)
        """
        try:
            reasoning_data = self._extract_reasoning(llm_output)
            if not reasoning_data:
                return 0.0
            
            total_steps = len(reasoning_data.get("steps", []))
            valid_steps = len(reasoning_data.get("valid_steps", []))
            
            if total_steps == 0:
                return 0.0
            
            return (valid_steps / total_steps) * 100.0
        except Exception as e:
            logger.error(f"Reasoning evaluation failed: {e}")
            return 0.0
    
    def _extract_reasoning(self, response: str) -> Dict[str, Any]:
        """
        Extract structured reasoning from response.
        
        Looks for patterns like:
        - JSON blocks with "reasoning" key
        - Sections marked ---REASONING---, ---LOGICAL FORMALIZATION---, etc.
        """
        reasoning_data = {"steps": [], "valid_steps": []}
        
        # Try to extract JSON
        json_blocks = self._extract_json_blocks(response)
        for block in json_blocks:
            try:
                data = json.loads(block)
                if "reasoning" in data:
                    reasoning_list = data["reasoning"]
                    if isinstance(reasoning_list, list):
                        reasoning_data["steps"].extend(reasoning_list)
            except json.JSONDecodeError:
                pass
        
        # Try to extract sections
        sections = self._extract_sections(response, ["REASONING", "LOGICAL FORMALIZATION", "VERIFICATION"])
        for section_content in sections:
            lines = section_content.strip().split('\n')
            reasoning_data["steps"].extend([l.strip() for l in lines if l.strip()])
        
        # Validate each step
        for step in reasoning_data["steps"]:
            if self._validate_reasoning_step(step):
                reasoning_data["valid_steps"].append(step)
        
        return reasoning_data
    
    @staticmethod
    def _extract_json_blocks(text: str) -> List[str]:
        """Extract JSON blocks from markdown."""
        import re
        matches = re.findall(r'```json\n(.*?)\n```', text, re.DOTALL)
        return matches
    
    @staticmethod
    def _extract_sections(text: str, section_names: List[str]) -> List[str]:
        """Extract named sections."""
        sections = []
        for name in section_names:
            try:
                start = text.find(f"---{name}---")
                if start >= 0:
                    start += len(f"---{name}---")
                    end = text.find("---", start)
                    if end == -1:
                        end = len(text)
                    sections.append(text[start:end])
            except Exception:
                pass
        return sections
    
    @staticmethod
    def _validate_reasoning_step(step: str) -> bool:
        """Heuristic validation: non-empty, contains evidence markers."""
        if not step or len(step) < 10:
            return False
        
        evidence_markers = ["because", "since", "due to", "based on", "→", "∧", "∨"]
        return any(marker in step.lower() for marker in evidence_markers)
    
    def count_reasoning_steps(self, response: str) -> Tuple[int, int]:
        """
        Count total and valid reasoning steps.
        
        Returns:
            (total_steps, valid_steps)
        """
        reasoning_data = self._extract_reasoning(response)
        return len(reasoning_data["steps"]), len(reasoning_data["valid_steps"])


class FactualityEvaluator(BaseMetricEvaluator):
    """
    Evaluates factuality via Natural Language Inference (NLI).
    
    Uses RoBERTa-based fact-checking model to verify if LLM claims
    are supported by expert reference (entailment = factual).
    """
    
    def __init__(self, model_name: str = "cross-encoder/qnli-distilroberta-base"):
        self.model_name = model_name
        try:
            from transformers import pipeline
            self.nli_pipeline = pipeline("zero-shot-classification", model=model_name)
            logger.info(f"FactualityEvaluator initialized with {model_name}")
        except Exception as e:
            logger.error(f"Failed to load factuality model: {e}")
            self.nli_pipeline = None
    
    def evaluate(self, llm_output: str, expert_reference: str) -> float:
        """
        Compute factuality score (0-1: refuted to entailed).
        
        Args:
            llm_output: LLM-generated text
            expert_reference: Expert reference text
        
        Returns:
            Factuality score (1.0 = fully supported by expert, 0.0 = contradicted)
        """
        if self.nli_pipeline is None:
            logger.warning("NLI pipeline unavailable. Returning dummy factuality score.")
            return 0.5
        
        try:
            # Template: does expert_reference support llm_output?
            result = self.nli_pipeline(
                expert_reference,
                [llm_output],
                hypothesis_template="{}",
            )
            
            # Extract entailment score (0=refuted, 1=entailed)
            scores = result.get("scores", [0.5])
            return np.mean(scores)
        except Exception as e:
            logger.error(f"Factuality evaluation failed: {e}")
            return 0.0


class SemanticAlignmentMetrics:
    """Composite metrics for semantic analysis."""
    
    def __init__(self):
        self.wasserstein = WassersteinAnalyzer()
        self.reasoning = ReasoningEvaluator()
        self.factuality = FactualityEvaluator()
    
    def compute_all_metrics(
        self,
        llm_output: str,
        expert_reference: str,
    ) -> Dict[str, float]:
        """
        Compute all semantic metrics.
        
        Returns:
            Dictionary with keys: wasserstein_distance, reasoning_percentage, factuality_score
        """
        return {
            "wasserstein_distance": self.wasserstein.evaluate(llm_output, expert_reference),
            "reasoning_percentage": self.reasoning.evaluate(llm_output, expert_reference),
            "factuality_score": self.factuality.evaluate(llm_output, expert_reference),
        }
