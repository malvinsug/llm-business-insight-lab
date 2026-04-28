# LLM Business Insight Lab

**High-Reliability BI Framework for KPI-Driven Analysis via Prompting Strategy Optimization**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository implements a **comprehensive evaluation framework** for analyzing AI-generated business insights. It combines multiple evaluation methodologies to assess the quality, factuality, and actionability of LLM responses against human expert benchmarks.

The framework evaluates the **BACoT (Business Analysis Corpus of Thought)** benchmark—80 real-world business scenarios across multiple departments—using:

- **Readability Metrics**: Flesch-Kincaid, Flesch Reading Ease
- **Semantic Similarity**: BERT-Score for contextual token-level matching
- **Factuality Verification**: NLI-based (using MiniCheck-DeBERTa-v3-Large)
- **Actionability Analysis**: Precision metrics for recommendation quality
- **Reasoning Evaluation**: Structured reasoning chain extraction and verification

Responses are evaluated against expert-provided reference answers, with results stored as structured pickle DataFrames for statistical analysis and comparison.

---

## Core Architecture

### Evaluation Pipeline

```
Question + Reference Answer
        ↓
[LLM Generate Response]
        ↓
[Extract Text Components]
  ├─ AITextExtractor
  │  ├─ Observations
  │  ├─ Hypotheses
  │  └─ Recommendations
  └─ HumanAnswerExtractor
     ├─ Observations
     ├─ Interpretations
     └─ Recommendations
        ↓
[Compute Metrics via MainMetrics]
  ├─ Readability
  │  ├─ Flesch-Kincaid Grade
  │  └─ Flesch Reading Ease
  ├─ Semantic Similarity
  │  └─ BERT-Score
  ├─ Factuality
  │  └─ NLIMiniCheckEvaluator (MiniCheck-DeBERTa)
  ├─ Actionability
  │  └─ NLI-based evaluation
  └─ Reasoning Quality
     └─ ReasoningEvaluator
        ↓
[Aggregate Results]
  └─ Save to DataFrame (pickle format)
        ↓
Export → CSV/JSON for analysis
```

### Key Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **AITextExtractor** | Parse AI-generated content | Raw LLM response | Structured components |
| **HumanAnswerExtractor** | Parse expert answers | Expert provided text | Sections/observations |
| **MainMetrics** | Compute all evaluation metrics | Text pair (AI, expert) | Metric scores (dict) |
| **NLIMiniCheckEvaluator** | Factuality checking | Claims + reference | Entailment scores (0-1) |
| **ReasoningEvaluator** | Reasoning quality | Response + question | Reasoning count + verification |

---

## Installation

### Prerequisites

- Python 3.10+
- macOS, Linux, or Windows with WSL2
- GPU recommended for embeddings (CUDA 11.8+ or MPS for Apple Silicon)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/llm-business-insight-lab.git
cd llm-business-insight-lab

# Create virtual environment
python3.10 -m venv lbil
source lbil/bin/activate  # or `lbil\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### API Keys

Set environment variables on `.env`.

---

## Quick Start

### 1. Run Metrics Evaluation

```bash
# Evaluate BACoT dataset with main metrics (Flesch-Kincaid, BERT-Score, Factuality, Actionability, Reasoning)
python runner_main_metrics.py
```

### 2. Expected Dataset

Results are stored as pickle files in `data/`:

```
data/
├── bacot_complete.pkl                   ← Full dataset with all metrics
├── bacot_main_metrics_ai.pkl             ← AI-generated metrics
├── bacot_main_metrics_human.pkl          ← Human expert metrics
├── bacot_metrics_normalized.pkl          ← Normalized metric scores
└── bacot_all_metrics.pkl
```

---

## Project Structure

```
llm-business-insight-lab/
├── pyproject.toml                 # Project configuration + dependencies
├── requirements.txt               # Pip requirements
├── README.md                      # This file
├── Makefile                       # Build and utility commands
│
├── runner_main_metrics.py         # Main evaluation runner (Flesch, BERT, Factuality, etc.)
├── playground.ipynb               # Interactive exploration notebook
│
├── src/
│   ├── __init__.py
│   ├── extractor/
│   │   ├── __init__.py
│   │   ├── ai.py                 # AITextExtractor: Extract observations, hypotheses, recommendations from AI responses
│   │   └── human.py              # HumanAnswerExtractor: Parse human expert answers
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── main_metrics.py        # MainMetrics: Flesch-Kincaid, Flesch Reading Ease, BERT-Score, Factuality, Actionability, Reasoning
│   │
│   ├── nli/
│   │   ├── __init__.py
│   │   └── minicheck.py           # NLIMiniCheckEvaluator: NLI-based factuality checking (MiniCheck model)
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── evaluator.py           # ReasoningEvaluator: Extract and evaluate reasoning chains
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py              # Utility functions and configuration
│
├── data/
│   ├── bacot_complete.pkl         # Full BACoT dataset with all results
│   ├── bacot_main_metrics_ai.pkl   # AI-generated response metrics
│   ├── bacot_main_metrics_human.pkl # Human expert metrics
│   └── bacot_all_metrics.pkl       # Comprehensive metrics suite
│
├── experiments/
│   └── main_experiment.ipynb      # Jupyter notebook for experiment exploration
│
└── tests/
    ├── __init__.py
    └── test_core.py               # Unit tests
```

---

## Core Modules

### 1. Extractors (`src/extractor/`)

**AITextExtractor** (`ai.py`)
- Extracts observations and hypotheses from AI-generated reasoning chains
- Extracts actionable recommendations from model outputs
- Structures unformed text into analyzable components

**HumanAnswerExtractor** (`human.py`)
- Parses expert human-provided answers
- Extracts sections: observations, interpretations, recommendations
- Validates section structure for consistency

### 2. Metrics (`src/metrics/`)

**MainMetrics** (`main_metrics.py`)
- **Flesch-Kincaid Grade**: Measures text readability complexity
- **Flesch Reading Ease**: General readability score (0-100)
- **BERT-Score**: Contextual token-level semantic similarity
- **Factuality Score**: NLI-based verification (uses MiniCheck model)
- **Actionability Score**: Measures how actionable recommendations are
- **Reasoning Evaluation**: Counts how many valid reasoning for each recommendations.

### 3. NLI Evaluation (`src/nli/`)

**NLIMiniCheckEvaluator** (`minicheck.py`)
- Wraps MiniCheck-DeBERTa-v3-Large model for Natural Language Inference
- Scores factuality as: 0 (refuted) → 0.5 (neutral) → 1 (entailed)
- Evaluates individual claims against reference documents
- Calculates factuality precision and recall

### 4. Reasoning Analysis (`src/reasoning/`)

**ReasoningEvaluator** (`evaluator.py`)
- Extracts reasoning chains from LLM responses
- Sends reasoning to evaluation LLM (Azure) for validation
- Parses JSON-structured reasoning outputs
- Counts total reasoning steps vs. valid verified steps

### 5. Utilities (`src/utils/`)

**Config** (`config.py`)
- Configuration management
- Helper functions for data processing

---

## Methodology

### Data Format

All data is stored in pickle format (`.pkl`) for efficient serialization:

```python
import pickle

# Load BACoT dataset with metrics
with open("data/bacot_complete.pkl", "rb") as f:
    df = pickle.load(f)  # Returns pandas DataFrame

# Available columns include:
# - question, ai_answer, human_answer
# - factuality_score, actionability_score, bert_score
# - flesch_kincaid, flesch_reading_ease
# - reasoning_evaluation results
```

### Metric Definitions

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Flesch-Kincaid Grade** | 0-16+ | Years of education needed; 8-9 is ideal for general audience |
| **Flesch Reading Ease** | 0-100 | Higher = easier to read; 60-70 is considered "standard" |
| **BERT-Score** | 0-1 | Contextual token-level semantic similarity |
| **Factuality Score** | 0-1 | 0=refuted, 1=entailed (via NLI) |
| **Actionability Score** | 0-1 | 0=refuted, 1=entailed (via NLI) |
| **Reasoning ** | 0-1 | The percentage of valid reasoning based on recommendations |

### Computation Pipeline

```
Input Question + Reference Answer
        ↓
[AI Model Generate Response]
        ↓
[Extract Components]
  - Observations
  - Recommendations
  - Reasoning Chains
        ↓
[Compute Metrics]
  - Readability (Flesch)
  - Semantic Similarity (BERT)
  - Factuality (NLI + MiniCheck)
  - Actionability (NLI-based)
  - Reasoning Quality
        ↓
[Save to Pickle DataFrame]
        ↓
Export → CSV or JSON for analysis
```

---

## Extending the Framework

### Adding a Custom Metric

```python
from src.metrics.main_metrics import MainMetrics

class CustomMetric:
    @staticmethod
    def evaluate_custom_metric(text: str) -> float:
        """
        Implement your custom evaluation logic.
        """
        # Your implementation
        return score_value
```

---

## Performance

Typical evaluation performance on standard hardware:

| Task |  Model |
|------|--------|
| Load BACoT dataset (80 samples) | Pickle binary |
| Compute Flesch metrics | Statistical |
| Compute BERT-Score | GPU |
| NLI Factuality eval | MiniCheck-DeBERTa |
| Reasoning evaluation | Azure OpenAI |
| **Total full evaluation ** | CPU/GPU mixed |

---

## Contributing

We welcome contributions! Areas for enhancement:

- [ ] Additional evaluation metrics 
- [ ] Multi-language support for NLI evaluation
- [ ] Streaming evaluation pipeline for real-time processing
- [ ] Dashboard/visualization tool for result analysis
- [ ] Benchmark against other factuality models (QuestionAnswering NLI, FactKG)
- [ ] Support for fine-tuned domain-specific evaluators
- [ ] Integration with vector databases for embedding analysis
- [ ] Parallel batch processing improvements

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact & Support

- **Questions**: Open an issue on GitHub
- **Thesis Collaboration**: Contact author at `malvin.sugiri@gmail.com`.
---
