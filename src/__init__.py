"""
LLM Business Insight Lab: Production-Grade Dual-Agent Framework for KPI Analysis

This package implements:
- Dual-Agent Architecture (Logic Analyst + Strategic Consultant)
- SymCoT (Symbolic Chain of Thought with FOL verification)
- NoCoT (Direct strategic synthesis)
- Wasserstein Distance-based semantic alignment metrics
- BACoT (Business Analysis Corpus of Thought) benchmark
"""

__version__ = "0.1.0"
__author__ = "Research Engineer"

from src.data_models import (
    ExperimentConfig,
    ExperimentResult,
    PromptStrategy,
    Department,
    Category,
)

from src.agents import (
    LogicAnalystAgent,
    StrategicConsultantAgent,
    DualAgentOrchestrator,
)

from src.metrics import (
    WassersteinAnalyzer,
    ReasoningEvaluator,
    FactualityEvaluator,
)

from src.evaluators import BACoTBenchmark

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "PromptStrategy",
    "Department",
    "Category",
    "LogicAnalystAgent",
    "StrategicConsultantAgent",
    "DualAgentOrchestrator",
    "WassersteinAnalyzer",
    "ReasoningEvaluator",
    "FactualityEvaluator",
    "BACoTBenchmark",
]
