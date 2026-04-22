"""Evaluators Module: BACoT benchmark and experiment orchestration.

Implements:
1. BACoTBenchmark: Manages the Business Analysis Corpus of Thought dataset
2. ExperimentRunner: Orchestrates execution of experiments with metrics tracking
3. ResultsAggregator: Collects and exports results to JSON/CSV for reporting
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from uuid import uuid4

from loguru import logger
import pandas as pd

from src.data_models import (
    ExperimentConfig,
    ExperimentResult,
    EvaluationMetrics,
    BACoTDataset,
    Department,
    Category,
    PromptStrategy,
)
from src.agents import DualAgentOrchestrator
from src.metrics import SemanticAlignmentMetrics


class BACoTBenchmark:
    """
    Business Analysis Corpus of Thought (BACoT) Benchmark
    
    Represents 80 real-world business scenarios across 4 departments,
    each annotated with expert gold-standard answers and KPI context.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize BACoT benchmark.
        
        Args:
            data_path: Path to BACoT dataset JSON (if None, uses placeholder)
        """
        self.data_path = data_path
        self.dataset: Optional[BACoTDataset] = None
        self.scenarios: List[Dict[str, Any]] = []
    
    def load_dataset(self) -> BACoTDataset:
        """
        Load BACoT dataset.
        
        Returns:
            BACoTDataset with 80 scenarios
        """
        if self.data_path and self.data_path.exists():
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                self.scenarios = data.get("scenarios", [])
                logger.info(f"Loaded {len(self.scenarios)} scenarios from {self.data_path}")
        else:
            logger.warning("BACoT dataset not found. Using placeholder scenarios.")
            self.scenarios = self._create_placeholder_scenarios()
        
        self.dataset = BACoTDataset(scenarios=self.scenarios)
        return self.dataset
    
    @staticmethod
    def _create_placeholder_scenarios() -> List[Dict[str, Any]]:
        """Create placeholder scenarios for demonstration."""
        scenarios = []
        for i in range(80):
            dept = [d.value for d in list(Department)][(i // 20) % 4]
            cat = [c.value for c in list(Category)][(i // 5) % 4]
            
            scenario = {
                "id": f"scenario_{i:03d}",
                "department": dept,
                "category": cat,
                "user_question": f"What is the trend in KPI_{i} for Q{(i % 4) + 1}?",
                "selected_kpis": [f"KPI_{i}", f"KPI_{i+1}"],
                "expert_answer": f"[Expert analysis for scenario {i}]",
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def get_scenarios_for_config(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """
        Filter scenarios based on experiment config.
        
        Args:
            config: Experiment configuration with department/category filters
        
        Returns:
            Filtered list of scenarios
        """
        filtered = self.scenarios
        
        # Department filter
        dept_values = [d.value for d in config.departments]
        filtered = [s for s in filtered if s.get("department") in dept_values]
        
        # Category filter
        cat_values = [c.value for c in config.categories]
        filtered = [s for s in filtered if s.get("category") in cat_values]
        
        # Size limit
        if config.n_samples:
            filtered = filtered[:config.n_samples]
        
        return filtered


class ExperimentRunner:
    """
    Orchestrates experiment execution with full metrics tracking.
    
    Workflow:
    1. Load BACoT scenarios
    2. Initialize agent orchestrator
    3. For each scenario:
       - Execute agents (SymCoT, NoCoT)
       - Compute all metrics
       - Store results
    4. Export results to CSV/JSON
    """
    
    def __init__(
        self,
        orchestrator: DualAgentOrchestrator,
        benchmark: BACoTBenchmark,
        output_dir: Path,
    ):
        self.orchestrator = orchestrator
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_computer = SemanticAlignmentMetrics()
        self.results: List[ExperimentResult] = []
        
        logger.info(f"ExperimentRunner initialized with output_dir={output_dir}")
    
    async def run_experiment(
        self,
        config: ExperimentConfig,
    ) -> List[ExperimentResult]:
        """
        Execute full experiment against BACoT.
        
        Args:
            config: Experiment configuration
        
        Returns:
            List of ExperimentResult
        """
        logger.info(f"Starting experiment: {config.model_name}, strategy={config.prompt_strategy.value}")
        
        # Load scenarios
        scenarios = self.benchmark.get_scenarios_for_config(config)
        logger.info(f"Running {len(scenarios)} scenarios")
        
        for i, scenario in enumerate(scenarios):
            logger.debug(f"Executing scenario {i+1}/{len(scenarios)}: {scenario['id']}")
            
            result = await self._run_scenario(config, scenario)
            if result:
                self.results.append(result)
        
        logger.info(f"Experiment complete. Collected {len(self.results)} results.")
        return self.results
    
    async def _run_scenario(
        self,
        config: ExperimentConfig,
        scenario: Dict[str, Any],
    ) -> Optional[ExperimentResult]:
        """Execute single scenario."""
        try:
            experiment_id = str(uuid4())
            
            # Prepare context
            context = {
                "kpis": scenario.get("selected_kpis", []),
                "historical_data": scenario.get("data_context", {}),
                "timeframe": scenario.get("timeframe", "Q1-Q4"),
            }
            
            # Run orchestrator (select strategy from config)
            target_strategies = [config.prompt_strategy]
            responses = await self.orchestrator.orchestrate(
                user_question=scenario["user_question"],
                context=context,
                config=config,
                target_strategies=target_strategies,
            )
            
            if not responses:
                logger.warning(f"No responses for scenario {scenario['id']}")
                return None
            
            # Take first response (matching the requested strategy)
            response = list(responses.values())[0]
            
            # Compute metrics
            metrics = self._compute_metrics(
                response.content,
                scenario.get("expert_answer", ""),
                response,
            )
            
            # Create result
            result = ExperimentResult(
                experiment_id=experiment_id,
                timestamp=datetime.utcnow(),
                config=config,
                department=Department[scenario.get("department", "DEVELOPMENT").upper()],
                category=Category[scenario.get("category", "KPI_TREND").upper()],
                scenario_id=scenario["id"],
                user_question=scenario["user_question"],
                selected_kpis=scenario.get("selected_kpis", []),
                model_answer=response.content,
                reasoning_tree=response.reasoning,
                expert_answer=scenario.get("expert_answer", ""),
                metrics=metrics,
                model_name=config.model_name,
                prompt_strategy=config.prompt_strategy.value,
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Scenario {scenario['id']} failed: {e}")
            return None
    
    def _compute_metrics(
        self,
        llm_output: str,
        expert_reference: str,
        response: Any,
    ) -> EvaluationMetrics:
        """Compute all metrics for a response."""
        # Semantic metrics
        semantic = self.metrics_computer.compute_all_metrics(llm_output, expert_reference)
        
        # Reasoning metrics (SymCoT only)
        if response.reasoning:
            total_steps, valid_steps = 0, 0
            # Count reasoning steps if available
        else:
            total_steps, valid_steps = 0, 0
        
        return EvaluationMetrics(
            factuality_score=semantic.get("factuality_score", 0.5),
            reasoning_count=valid_steps,
            reasoning_percentage=(valid_steps / max(total_steps, 1)) * 100,
            wasserstein_distance=semantic.get("wasserstein_distance", 0.0),
            semantic_similarity=1.0 - (semantic.get("wasserstein_distance", 0.0) / 10),  # Normalize
            latency_ms=response.latency_ms,
        )
    
    def export_results_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export results to CSV (LinkedIn-ready).
        
        Returns:
            Path to exported CSV
        """
        if not self.results:
            logger.warning("No results to export")
            return None
        
        filename = filename or f"experiment_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.output_dir / filename
        
        rows = [r.to_csv_row() for r in self.results]
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Results exported to {filepath}")
        return filepath
    
    def export_results_json(self, filename: Optional[str] = None) -> Path:
        """
        Export results to JSON (full fidelity).
        
        Returns:
            Path to exported JSON
        """
        if not self.results:
            logger.warning("No results to export")
            return None
        
        filename = filename or f"experiment_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        data = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_results": len(self.results),
                "model": self.results[0].model_name if self.results else "unknown",
            },
            "results": [json.loads(r.model_dump_json()) for r in self.results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")
        return filepath
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary statistics for reporting.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame([r.to_csv_row() for r in self.results])
        
        summary = {
            "total_scenarios": len(self.results),
            "avg_factuality": float(df["factuality_score"].mean()),
            "avg_wasserstein": float(df["wasserstein_distance"].mean()),
            "avg_latency_ms": float(df["latency_ms"].mean()),
            "by_department": df.groupby("department").agg({
                "factuality_score": "mean",
                "wasserstein_distance": "mean",
                "latency_ms": "mean",
            }).to_dict(),
        }
        
        logger.info(f"Summary Report:\n{json.dumps(summary, indent=2)}")
        return summary


class ResultsAggregator:
    """Aggregates and analyzes results across multiple experiments."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
    
    def load_all_csv_results(self) -> pd.DataFrame:
        """Load all CSV results from directory."""
        csv_files = list(self.results_dir.glob("**/*.csv"))
        dfs = [pd.read_csv(f) for f in csv_files]
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total results from {len(csv_files)} files")
            return combined
        
        return pd.DataFrame()
    
    def compare_strategies(self) -> Dict[str, Any]:
        """Compare performance across prompt strategies."""
        df = self.load_all_csv_results()
        
        if df.empty:
            return {}
        
        return {
            "by_strategy": df.groupby("prompt_strategy").agg({
                "factuality_score": ["mean", "std"],
                "wasserstein_distance": ["mean", "std"],
                "latency_ms": ["mean", "std"],
            }).to_dict(),
        }
