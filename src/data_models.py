"""
Data Models: Core Pydantic schemas for experiment configuration, results, and metrics.

This module defines immutable, type-safe data structures for the entire LLM Business Insight Lab,
ensuring high-signal experiment tracking from hypothesis to evaluation.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field, validator


class PromptStrategy(str, Enum):
    """Prompting strategies evaluated in this framework."""
    NOCOT = "nocot"  # Direct strategic synthesis without structured reasoning
    COT = "cot"      # Standard Chain-of-Thought
    SYMCOT = "symcot"  # Symbolic CoT with First-Order Logic verification


class Department(str, Enum):
    """Business departments covered in BACoT benchmark."""
    DEVELOPMENT = "development"
    SALES = "sales"
    CUSTOMER_CARE = "customer_care"
    MARKETING = "marketing"


class Category(str, Enum):
    """Analysis categories within each department."""
    KPI_TREND = "kpi_trend_and_performance"
    CUSTOMER_INSIGHTS = "customer_insights_and_impact"
    PROCESS_EFFICIENCY = "process_efficiency"
    FORECASTING = "forecasting_and_strategic_planning"


class ExperimentConfig(BaseModel):
    """
    Experiment configuration: defines all parameters for a benchmark run.
    
    Mathematical Notes:
    - temperature=1.0 maintains "deterministic creativity": fixed seed ensures reproducibility
      while allowing enough variance for diverse hypothesis generation.
    - n_samples: BACoT contains 80 real-world scenarios; recommend full coverage (n_samples=80)
      for production deployments.
    """
    model_name: str = Field(..., description="LLM model identifier (e.g., 'gpt-4', 'claude-opus')")
    prompt_strategy: PromptStrategy = Field(..., description="Prompting strategy to evaluate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="LLM temperature (fixed for determinism)")
    seed: int = Field(42, description="Random seed for reproducibility")
    n_samples: Optional[int] = Field(None, description="Number of BACoT samples to evaluate (None=all)")
    departments: List[Department] = Field(default_factory=lambda: list(Department), description="Departments to evaluate")
    categories: List[Category] = Field(default_factory=lambda: list(Category), description="Categories to evaluate")
    max_tokens: int = Field(2048, ge=256, le=4096, description="Max tokens per LLM response")
    timeout_seconds: float = Field(30.0, ge=5.0, description="API timeout per request")
    
    class Config:
        validate_assignment = True


class EvaluationMetrics(BaseModel):
    """
    Quantitative metrics for a single LLM response against expert gold standard.
    
    Mathematical Definitions:
    - factuality_score: Binary classification via NLI (supports=1, refutes=0)
    - reasoning_count: Number of valid First-Order Logic reasoning steps extracted
    - wasserstein_distance: 2-Wasserstein distance between LLM and expert embeddings
      (lower = better semantic alignment; range [0, ∞))
    - latency_ms: Wall-clock time for LLM generation
    """
    factuality_score: float = Field(..., ge=0.0, le=1.0, description="Factuality (NLI-based)")
    reasoning_count: int = Field(..., ge=0, description="Valid reasoning steps in SymCoT")
    reasoning_percentage: float = Field(..., ge=0.0, le=100.0, description="% of reasonings that were valid")
    wasserstein_distance: float = Field(..., ge=0.0, description="2-Wasserstein distance to expert answer")
    semantic_similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity with expert")
    latency_ms: float = Field(..., ge=0.0, description="LLM generation latency in milliseconds")
    clarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Clarity assessment (if computed)")


class ExperimentResult(BaseModel):
    """
    Complete result of a single experiment: configuration + response + metrics.
    
    This replaces scattered pickle/CSV files in the original thesis work, enabling:
    1. Type-safe serialization via Pydantic's model_dump()
    2. Batch export to JSON/CSV for LinkedIn dashboards
    3. Audit trail: config + prompt + response + metrics in one record
    """
    experiment_id: str = Field(..., description="Unique identifier (UUID4)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Experiment execution time")
    
    # Configuration
    config: ExperimentConfig = Field(..., description="Experiment hyperparameters")
    
    # Business Context
    department: Department = Field(..., description="Business unit")
    category: Category = Field(..., description="Analysis category")
    scenario_id: str = Field(..., description="BACoT scenario ID")
    
    # Input
    user_question: str = Field(..., description="Original KPI question from user")
    selected_kpis: List[str] = Field(..., description="KPIs mentioned in question")
    
    # LLM Output
    model_answer: str = Field(..., description="LLM-generated response")
    reasoning_tree: Optional[Dict[str, Any]] = Field(None, description="Parsed reasoning (SymCoT only)")
    
    # Expert Reference
    expert_answer: str = Field(..., description="Gold-standard expert response")
    
    # Metrics
    metrics: EvaluationMetrics = Field(..., description="Evaluation results")
    
    # Metadata
    model_name: str = Field(..., description="Cached from config for CSV export")
    prompt_strategy: str = Field(..., description="Cached from config for CSV export")
    
    class Config:
        validate_assignment = True
    
    def to_csv_row(self) -> Dict[str, Any]:
        """
        Flatten ExperimentResult to CSV-exportable row.
        
        Returns:
            Dictionary with flattened keys suitable for pd.DataFrame(rows)
        """
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model_name,
            "prompt_strategy": self.prompt_strategy,
            "department": self.department.value,
            "category": self.category.value,
            "scenario_id": self.scenario_id,
            "latency_ms": self.metrics.latency_ms,
            "factuality_score": self.metrics.factuality_score,
            "reasoning_count": self.metrics.reasoning_count,
            "reasoning_percentage": self.metrics.reasoning_percentage,
            "wasserstein_distance": self.metrics.wasserstein_distance,
            "semantic_similarity": self.metrics.semantic_similarity,
            "clarity_score": self.metrics.clarity_score or None,
        }


@dataclass
class StratificationConfig:
    """Configuration for stratified analysis splits (as in original thesis work)."""
    department: Department
    category: Category
    n_samples_per_stratum: int = 10
    random_state: int = 42
    
    def to_stratum_key(self) -> str:
        """Generates a key for data dictionaries (e.g., 'development_kpi_trend')."""
        return f"{self.department.value}_{self.category.value}"


@dataclass
class BACoTDataset:
    """BACoT (Business Analysis Corpus of Thought) benchmark dataset structure."""
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    timestamp_created: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    total_scenarios: int = 80  # Real-world business scenarios
    
    def filter_by_department(self, dept: Department) -> List[Dict[str, Any]]:
        """Filter scenarios by department."""
        return [s for s in self.scenarios if s.get("department") == dept.value]
    
    def filter_by_category(self, cat: Category) -> List[Dict[str, Any]]:
        """Filter scenarios by category."""
        return [s for s in self.scenarios if s.get("category") == cat.value]
