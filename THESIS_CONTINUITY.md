# Thesis Continuity & Research Bridge

This repository extends the M.Sc. thesis research from TU Munich on **"Prompting Strategies for KPI-Driven Business Insights"** into a production-grade, modular framework.

## Key Thesis Findings → Framework Implementation

### 1. The "Guidance Bias" Paradox

**Thesis Finding**: CoT (Complex reasoning chains) underperforms compared to NoCoT (direct synthesis) for strategic C-level tasks.

**Implementation**:
- `LogicAnalystAgent`: Reserved for rigor-driven tasks (engineering, RCA)
- `StrategicConsultantAgent`: Optimal for executive briefings and strategic planning
- `DualAgentOrchestrator`: Enables side-by-side comparison

**Relevance**: Challenges conventional wisdom and provides data-backed evidence for Singaporean CTOs to make informed prompting choices.

### 2. First-Order Logic (FOL) Verification in SymCoT

**Thesis Finding**: Symbolic Chain of Thought with FOL achieved highest factuality scores by verifying logical consistency before generating recommendations.

**Implementation**:
```python
class LogicAnalystAgent(BaseAgent):
    """
    Formalize observations as First-Order Logic predicates:
    P(kpi_x, quarter_y, trend_z) → R(action_required)
    Verify logical consistency before output.
    """
```

**Benefit**: Reduces hallucinations, ensuring recommendations are grounded in data.

### 3. Wasserstein Distance for Semantic Alignment

**Thesis Finding**: Traditional metrics (ROUGE, BLEU) fail to capture semantic nuance. PCA-based 2-Wasserstein distance shows strong alignment with human judgment.

**Implementation**:
```python
class WassersteinAnalyzer(BaseMetricEvaluator):
    """
    W₂(LLM_embeddings, Expert_embeddings)
    = [min_{π ∈ Π} E[||X - Y||²]]^(1/2)
    """
```

**Advantage**: Moves beyond surface-level similarity to measure true semantic alignment.

### 4. BACoT (Business Analysis Corpus of Thought)

**Thesis Finding**: Synthetic benchmarks (e.g., GSM8K) don't capture real business context. Developed BACoT with 80 real-world scenarios.

**Implementation**:
- `BACoTBenchmark` class loads 80 scenarios across 4 departments
- Stratified evaluation by Department × Category
- Expert gold-standard answers for comparison

**Real-World Grounding**: Ensures insights are relevant to actual CTO decision-making.

### 5. Departmental Optimization

**Thesis Finding**: No single prompting strategy is universally optimal. Effectiveness is highly department-dependent.

**Implementation**:
- Marketing & Strategic Planning → NoCoT (better for foresight)
- Development & Engineering → SymCoT (high factuality required)
- Customer Care → Best hybrid approach (balance)

**Framework Support**:
```python
config = ExperimentConfig(
    departments=[Department.DEVELOPMENT, Department.SALES],
    categories=[Category.KPI_TREND],  # Controlled evaluation
)
```

### 6. Temperature & Deterministic Creativity

**Thesis Finding**: Temperature = 1.0 with fixed seed balances creativity for hypothesis generation with reproducibility for production audits.

**Implementation**:
```python
ExperimentConfig(
    temperature=1.0,  # Fixed for "deterministic creativity"
    seed=42,          # Reproducible randomness
)
```

---

## Refactoring: Academic → Production

### Original Thesis Structure **→** Modern Framework

| Aspect | Thesis Notebooks | Framework |
|--------|------------------|-----------|
| **Agents** | Ad-hoc LLM calls in cells | `LogicAnalystAgent`, `StrategicConsultantAgent` classes |
| **Metrics** | Scattered computation | `WassersteinAnalyzer`, `FactualityEvaluator` modules |
| **Data** | Pickles + CSVs | Pydantic `ExperimentResult` (type-safe, serializable) |
| **Reproducibility** | Manual pandas groupby | `StratificationConfig` + `ExperimentRunner` |
| **Export** | Ad-hoc CSV writes | `export_results_csv()`, `export_results_json()` |
| **Logging** | Print statements | Structured logging via `loguru` |

### Technology Stack Upgrade

```
Thesis Infrastructure          Framework Infrastructure
├─ Jupyter notebooks          └─ Async-first Python modules
├─ Pandas groupby             └─ Pydantic validation
├─ POT library (raw)          └─ Wrapped in WassersteinAnalyzer
├─ Transformers (raw)         └─ SemanticAlignmentMetrics
└─ Manual JSON exports        └─ Structured ResultsAggregator
```

---

## Key Capabilities Retained from Thesis

1. ✅ **Wasserstein Distance Computation**
   - Uses POT library with Earth Mover's Distance (EMD)
   - PCA reduction for interpretability (307 components default)

2. ✅ **Reasoning Quality Metrics**
   - Parses JSON from SymCoT responses
   - Validates logical consistency
   - Computes percentage of valid reasoning steps

3. ✅ **BACoT Benchmark**
   - All 80 real-world scenarios structurally preserved
   - Stratification by department and category

4. ✅ **Factuality via NLI**
   - RoBERTa-based classification (entailment = factual)
   - Direct port from thesis evaluation code

---

## New Capabilities Introduced

1. 🆕 **Agentic Framework**
   - Dual-agent orchestration via `DualAgentOrchestrator`
   - Async execution for parallel strategy evaluation

2. 🆕 **Type Safety**
   - Full Pydantic validation of configs and results
   - Eliminates silent data errors from thesis

3. 🆕 **LinkedIn-Ready Exports**
   - CSV format optimized for stakeholder dashboards
   - JSON preserves full fidelity for audit trails

4. 🆕 **Reproducible Deployments**
   - `pyproject.toml` for dependency pinning
   - Comprehensive logging via `loguru`

5. 🆕 **Testing & CI/CD**
   - Pytest suite for regression testing
   - Type checking via mypy

---

## From Thesis Writing to Content Strategy

The current framework directly supports the **"NoCoT vs. CoT"** LinkedIn narrative:

**Post Angle**: 
> "We evaluated 80 real business scenarios. Surprisingly, **NoCoT (direct synthesis) outperformed complex CoT chains for strategic planning**—contradicting conventional ML wisdom. Here's what leading CTOs should know..."

**Data Points from Framework**:
- Factuality scores by strategy (CSV export)
- Latency comparison (why NoCoT is faster)
- Department-specific recommendations (departmental analysis)
- Wasserstein alignment metrics (semantic proof)

---

## Running Original Thesis Experiments

To reproduce thesis figures/tables:

```python
# Load original BACoT scenarios
benchmark = BACoTBenchmark(data_path="data/bacot_dataset.json")

# Compare all three strategies
for strategy in [PromptStrategy.NOCOT, PromptStrategy.COT, PromptStrategy.SYMCOT]:
    config = ExperimentConfig(
        model_name="gpt-4",
        prompt_strategy=strategy,
        n_samples=80,  # Full BACoT
    )
    runner = ExperimentRunner(orchestrator, benchmark, output_dir)
    results = await runner.run_experiment(config)
    
    # Export for thesis analysis
    runner.export_results_csv(f"thesis_comparison_{strategy.value}.csv")
```

---

## Future Extensions (Beyond Thesis Scope)

1. **Multi-hop reasoning**: Chain multiple agents for complex analyses
2. **Fine-tuned models**: Task-specific LoRA adapters
3. **Real-time dashboards**: Streamlit front-end for CTOs
4. **Cost optimization**: Track API spend per strategy
5. **Custom domains**: Extend beyond KPI analysis (compliance, technical debt, etc.)

---

**Bottom Line**: This framework is the **production sibling** of your thesis—same rigor, scientific grounding, and insights, but architected for scaling, reproducibility, and enterprise deployment.

---

*Last Updated: April 2025 | Framework Version: 0.1.0 | Thesis: "Prompting Strategies for KPI-Driven Business Insights" (TU Munich, 2024)*
