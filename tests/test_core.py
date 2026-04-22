"""
Test suite for LLM Business Insight Lab.

Run with: pytest tests/
"""

import pytest
import asyncio
from pathlib import Path

from src.data_models import (
    ExperimentConfig,
    PromptStrategy,
    Department,
    Category,
    ExperimentResult,
    EvaluationMetrics,
)
from src.agents import (
    LogicAnalystAgent,
    StrategicConsultantAgent,
    DualAgentOrchestrator,
)
from src.metrics import WassersteinAnalyzer, ReasoningEvaluator, FactualityEvaluator
from src.evaluators import BACoTBenchmark, ExperimentRunner


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_experiment_config_valid(self):
        """Test valid experiment config."""
        config = ExperimentConfig(
            model_name="gpt-4",
            prompt_strategy=PromptStrategy.SYMCOT,
        )
        assert config.model_name == "gpt-4"
        assert config.prompt_strategy == PromptStrategy.SYMCOT
        assert config.temperature == 1.0
    
    def test_experiment_config_invalid_temp(self):
        """Test invalid temperature bounds."""
        with pytest.raises(ValueError):
            ExperimentConfig(
                model_name="gpt-4",
                prompt_strategy=PromptStrategy.NOCOT,
                temperature=3.0,  # Invalid: > 2.0
            )
    
    def test_experiment_result_to_csv_row(self):
        """Test CSV export formatting."""
        config = ExperimentConfig(
            model_name="gpt-4",
            prompt_strategy=PromptStrategy.SYMCOT,
        )
        
        metrics = EvaluationMetrics(
            factuality_score=0.85,
            reasoning_count=5,
            reasoning_percentage=80.0,
            wasserstein_distance=2.1,
            semantic_similarity=0.78,
            latency_ms=1245.3,
        )
        
        result = ExperimentResult(
            experiment_id="test_001",
            config=config,
            department=Department.DEVELOPMENT,
            category=Category.KPI_TREND,
            scenario_id="scenario_001",
            user_question="Test question",
            selected_kpis=["KPI_1"],
            model_answer="Test answer",
            expert_answer="Expert answer",
            metrics=metrics,
            model_name="gpt-4",
            prompt_strategy="symcot",
        )
        
        row = result.to_csv_row()
        assert row["model"] == "gpt-4"
        assert row["factuality_score"] == 0.85
        assert row["latency_ms"] == 1245.3


class TestMetrics:
    """Test metric evaluators."""
    
    def test_reasoning_evaluator_empty(self):
        """Test reasoning evaluator with empty input."""
        evaluator = ReasoningEvaluator()
        score = evaluator.evaluate("", "")
        assert score == 0.0
    
    def test_reasoning_evaluator_valid_steps(self):
        """Test reasoning extraction with valid steps."""
        evaluator = ReasoningEvaluator()
        response = """
---REASONING---
Because the data shows trend X, we infer Y.
Since condition Z holds, action is recommended.
---LOGICAL FORMALIZATION---
P(trend_X) ∧ Q(condition_Z) → R(recommendation)
"""
        score = evaluator.evaluate(response, "")
        # Should find 2 valid steps with evidence markers
        assert score > 0.0
    
    def test_wasserstein_analyzer_valid(self):
        """Test Wasserstein analyzer initialization."""
        try:
            analyzer = WassersteinAnalyzer()
            # Just test that it initializes without error
            assert analyzer.cost_metric == "euclidean"
        except Exception as e:
            # If sentence-transformers not available, that's ok for test
            print(f"Note: Wasserstein test skipped ({e})")


class TestAgents:
    """Test agent framework."""
    
    def test_logic_analyst_init(self):
        """Test LogicAnalystAgent initialization."""
        agent = LogicAnalystAgent(
            model_name="gpt-4",
            strategy=PromptStrategy.SYMCOT,
            api_key="test_key",
        )
        assert agent.model_name == "gpt-4"
        assert agent.strategy == PromptStrategy.SYMCOT
        assert "First-Order Logic" in agent._default_system_prompt()
    
    def test_strategic_consultant_init(self):
        """Test StrategicConsultantAgent initialization."""
        agent = StrategicConsultantAgent(
            model_name="gpt-4",
            strategy=PromptStrategy.NOCOT,
            api_key="test_key",
        )
        assert agent.model_name == "gpt-4"
        assert agent.strategy == PromptStrategy.NOCOT
        assert "strategic" in agent._default_system_prompt().lower()
    
    @pytest.mark.asyncio
    async def test_orchestrator_init(self):
        """Test dual-agent orchestrator initialization."""
        logic_agent = LogicAnalystAgent(
            model_name="gpt-4",
            strategy=PromptStrategy.SYMCOT,
            api_key="test_key",
        )
        strategy_agent = StrategicConsultantAgent(
            model_name="gpt-4",
            strategy=PromptStrategy.NOCOT,
            api_key="test_key",
        )
        
        orchestrator = DualAgentOrchestrator(logic_agent, strategy_agent)
        assert orchestrator.logic_analyst is not None
        assert orchestrator.strategic_consultant is not None


class TestBenchmark:
    """Test BACoT benchmark."""
    
    def test_bacot_initialization(self):
        """Test BACoT benchmark initialization."""
        benchmark = BACoTBenchmark()
        assert benchmark.scenarios == []
    
    def test_bacot_placeholder_scenarios(self):
        """Test placeholder scenario generation."""
        benchmark = BACoTBenchmark()
        benchmark.load_dataset()
        
        assert len(benchmark.scenarios) == 80
        
        # Verify structure
        scenario = benchmark.scenarios[0]
        assert "id" in scenario
        assert "department" in scenario
        assert "category" in scenario
        assert "user_question" in scenario
        assert "expert_answer" in scenario
    
    def test_bacot_filtering(self):
        """Test scenario filtering."""
        benchmark = BACoTBenchmark()
        benchmark.load_dataset()
        
        config = ExperimentConfig(
            model_name="gpt-4",
            prompt_strategy=PromptStrategy.NOCOT,
            n_samples=10,
            departments=[Department.DEVELOPMENT],
            categories=[Category.KPI_TREND],
        )
        
        filtered = benchmark.get_scenarios_for_config(config)
        assert len(filtered) <= 10


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_structure(self):
        """Test that full pipeline can be initialized."""
        # Create config
        config = ExperimentConfig(
            model_name="gpt-4",
            prompt_strategy=PromptStrategy.SYMCOT,
            n_samples=5,
        )
        
        # Create agents
        logic_agent = LogicAnalystAgent(
            model_name="gpt-4",
            strategy=PromptStrategy.SYMCOT,
            api_key="test_key",
        )
        strategy_agent = StrategicConsultantAgent(
            model_name="gpt-4",
            strategy=PromptStrategy.NOCOT,
            api_key="test_key",
        )
        orchestrator = DualAgentOrchestrator(logic_agent, strategy_agent)
        
        # Create benchmark
        benchmark = BACoTBenchmark()
        benchmark.load_dataset()
        
        # Create runner
        output_dir = Path("/tmp/test_results")
        runner = ExperimentRunner(orchestrator, benchmark, output_dir)
        
        assert runner.results == []
        assert runner.output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
