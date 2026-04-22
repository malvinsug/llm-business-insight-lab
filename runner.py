"""
runner.py: Main experiment orchestration script.

Executes full benchmark: agents, evaluation, metrics tracking, and results export.

Usage:
    python runner.py --model gpt-4 --strategy symcot --n_samples 80
    python runner.py --config config.yaml
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime
import os

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)
logger.add(
    "logs/experiment_{time}.log",
    rotation="500 MB",
    retention="10 days",
)

from src.data_models import (
    ExperimentConfig,
    PromptStrategy,
    Department,
    Category,
)
from src.agents import (
    LogicAnalystAgent,
    StrategicConsultantAgent,
    DualAgentOrchestrator,
)
from src.evaluators import ExperimentRunner, BACoTBenchmark, ResultsAggregator


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Build experiment config
    config = ExperimentConfig(
        model_name=args.model,
        prompt_strategy=PromptStrategy(args.strategy),
        temperature=args.temperature,
        n_samples=args.n_samples,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    
    logger.info(f"Config: {config}")
    
    # Initialize agents
    logger.info("Initializing agents...")
    api_key = os.environ.get("OPENAI_API_KEY", "mock_key")
    
    logic_analyst = LogicAnalystAgent(
        model_name=config.model_name,
        strategy=PromptStrategy.SYMCOT,
        api_key=api_key,
    )
    
    strategic_consultant = StrategicConsultantAgent(
        model_name=config.model_name,
        strategy=PromptStrategy.NOCOT,
        api_key=api_key,
    )
    
    orchestrator = DualAgentOrchestrator(logic_analyst, strategic_consultant)
    
    # Load benchmark
    logger.info("Loading BACoT benchmark...")
    benchmark = BACoTBenchmark(
        data_path=Path(args.bacot_path) if args.bacot_path else None
    )
    benchmark.load_dataset()
    
    # Run experiment
    logger.info("Starting experiment runner...")
    runner = ExperimentRunner(orchestrator, benchmark, output_dir)
    results = await runner.run_experiment(config)
    
    # Export results
    logger.info("Exporting results...")
    csv_path = runner.export_results_csv()
    json_path = runner.export_results_json()
    
    summary = runner.generate_summary_report()
    
    # Save summary
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results: {csv_path}")
    logger.info(f"Full Data: {json_path}")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Total Scenarios: {summary.get('total_scenarios', 0)}")
    logger.info(f"  Avg Factuality: {summary.get('avg_factuality', 0):.3f}")
    logger.info(f"  Avg Wasserstein Distance: {summary.get('avg_wasserstein', 0):.3f}")
    logger.info(f"  Avg Latency: {summary.get('avg_latency_ms', 0):.1f}ms")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Business Insight Lab: Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SymCoT strategy with 80 BACoT scenarios
  python runner.py --model gpt-4 --strategy symcot --n_samples 80
  
  # Run NoCoT with limited samples for quick testing
  python runner.py --model gpt-4 --strategy nocot --n_samples 5
  
  # Run with custom temperature for increased creativity
  python runner.py --model gpt-4 --strategy nocot --temperature 1.5
        """,
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="LLM model (e.g., gpt-4, claude-opus, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--strategy",
        choices=["nocot", "cot", "symcot"],
        default="symcot",
        help="Prompting strategy to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM temperature (for deterministic creativity)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of BACoT scenarios to run (None = all 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens per LLM response",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--bacot_path",
        default=None,
        help="Path to BACoT dataset JSON (optional)",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
