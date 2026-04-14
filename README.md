# LLM Text Classification

**Bachelor thesis for evaluating Large Language Models (LLMs) on educational chatbot interaction classification.**

This project compares three LLMs (GPT-4o, Claude Sonnet, LLaMA 3) in classifying one-turn chatbot interactions across 6 dialogue act categories with role-based filtering and dual prompt strategies (basic + few-shot).

---

## 📚 Quick Start

### Installation & Setup

```bash
# Activate conda environment
conda activate algorithms

# Run a quick test (GPT-4 only, 2 messages per category)
python -m src.experiments.runner --test

# Run with custom options
python -m src.experiments.runner --limit 5 --categories response_substance --models gpt4o
```

### Output

Results are saved to `src/results/raw/results_TIMESTAMP.json` with:
- Per-message predictions
- True labels vs predicted labels
- Match indicators
- Model, strategy, category, and role metadata

Logs are saved to `logs/experiment_TIMESTAMP.log`

---

## 🏗️ Architecture Overview

The pipeline follows a modular, layered architecture:

```
RUNNER (CLI & validation)
    ↓
EXPERIMENT (Orchestrator)
    ↓
PIPELINE
    ├── Filter (role + text length validation)
    ├── Classifier (LLM calls)
    └── Result Builder (JSON persistence)
    ↓
STATISTICS & VIEWS (Reporting)
```

### Key Components

| Module | Purpose |
|--------|---------|
| `experiments/` | Orchestration and configuration |
| `pipeline/` | Core classification logic |
| `llm/` | LLM integrations (GPT-4o, Claude, LLaMA3) |
| `data/` | Data loading and models |
| `evaluation/` | Metrics and result analysis |
| `views/` | Console output formatting |
| `utils/` | Logging and error handling |
| `prompts/` | Prompt templates per category |

---

## 🔧 Recent Improvements (April 2026)

### 1. **Logging System** (`src/utils/logger.py`)
- Dual output: console + file logging
- DEBUG info in file, INFO in console
- Automatic log file generation with timestamps

### 2. **Error Handling** (`src/utils/error_handler.py`)
- Custom exception types: `DataLoadError`, `ClassificationError`, `ConfigError`
- `@handle_errors` decorator for automatic error logging
- `validate_or_error()` for configuration validation

### 3. **Enhanced Configuration** (`src/experiments/config.py`)
- Environment variable support (`DATA_DIR`, paths)
- Centralized path management
- Automatic config validation before experiment

### 4. **UI/View Separation** (`src/views/builder.py`)
- `ExperimentViewBuilder` handles ALL console output
- Separated presentation logic from experiment logic
- Easy to modify output format or create alternative views

### 5. **Pipeline Architecture**
- Moved classification loop to `ClassificationPipeline.classify_category()`
- Clean separation: orchestration vs execution
- Experiment.py now focuses on loop control

### 6. **Type Hints & Documentation**
- Added full type hints throughout
- Comprehensive docstrings
- Clear parameter descriptions

---

## 📋 Complete Project Structure

```
src/
├── experiments/              # Orchestration layer
│   ├── runner.py            # CLI entry point
│   ├── config.py            # Config + validation
│   ├── experiment.py        # Main orchestrator
│   └── stats.py             # Statistics tracking
│
├── pipeline/                # Core processing
│   ├── filter.py            # Message filtering (role + length)
│   ├── matcher.py           # Ground truth matching
│   ├── classifier_pipeline.py # Classification orchestration
│   └── result_builder.py    # JSON result persistence
│
├── llm/                     # LLM Integrations
│   ├── gpt_4o.py           # OpenAI GPT-4o
│   ├── claude_sonnet.py    # Anthropic Claude
│   ├── llama_3.py          # Ollama LLaMA 3
│   └── prompt_loader.py    # Prompt template loading
│
├── data/                    # Data handling
│   ├── loaders/
│   │   ├── interaction_loader.py
│   │   ├── ground_truth_loader.py
│   │   └── batch_iterator.py
│   └── models/
│       └── data_models.py   # Dataclasses (Message, PredictionResult, etc.)
│
├── evaluation/              # Analysis & metrics
│   ├── metrics.py          # Accuracy, F1, classification reports
│   └── run_results.py      # Mini-test validation
│
├── views/                   # Presentation layer
│   └── builder.py          # ExperimentViewBuilder (console output)
│
├── utils/                   # Utilities
│   ├── logger.py           # Logging setup
│   ├── error_handler.py    # Exception handling
│   └── __init__.py         # Package exports
│
├── prompts/                # Prompt templates
│   ├── cps_behaviour/
│   ├── interactional_move/
│   ├── is_followup/
│   ├── prompt_type/
│   ├── response_stance/
│   └── response_substance/
│
└── results/
    ├── raw/                # Experiment results
    └── test/               # Mini-test results
```

---

## 🚀 Usage Examples

### Quick Test (2 messages, GPT-4 only)
```bash
python -m src.experiments.runner --test
```

### Test Specific Category (5 messages, all LLMs)
```bash
python -m src.experiments.runner --limit 5 --categories response_substance --models gpt4o claude llama3
```

### Full Experiment (all messages, all categories, all models)
```bash
python -m src.experiments.runner
```

### Analyze Results
```bash
python -m src.evaluation.metrics --results src/results/raw/results_TIMESTAMP.json
```

### Available Options

| Flag | Example | Description |
|------|---------|-------------|
| `--test` | `--test` | Quick test mode (GPT-4, limit=2) |
| `--limit` | `--limit 10` | Max messages per category |
| `--models` | `--models gpt4o claude` | Which models to test |
| `--categories` | `--categories response_stance` | Which categories to test |
| `--interactions` | `--interactions data/...json` | Custom interactions path |
| `--ground-truth` | `--ground-truth data/...json` | Custom ground truth path |

---

## 📊 Categories & Strategies

### Dialogue Act Categories (6 total)

| Category | Role Filter | Description |
|----------|-------------|-------------|
| `response_stance` | Chatbot (1) | How chatbot responds (none/confident/neutral) |
| `prompt_type` | Teacher (0) | Type of teacher prompt (solution_request, elaborated_request, etc.) |
| `interactional_move` | Chatbot (1) | Chatbot's interactive move (probe_more_info, reflection, etc.) |
| `cps_behavior` | Teacher (0) | Teacher's complex problem-solving behavior |
| `response_substance` | Chatbot (1) | Substance of chatbot's response (none/elaborate/refines_previous) |
| `is_followup` | Both (None) | Whether message is a follow-up question |

### Prompt Strategies

- **basic**: Simple prompt without examples
- **few_shot**: Prompt with few-shot examples for in-context learning

---

## 🔍 Configuration

Environment variables (optional):
```bash
export DATA_DIR="/custom/data/path"
```

Default paths (in `config.py`):
- Interactions: `data/process_data/processed_interactions.json`
- Ground truth: `data/process_data/processed_ground_truths.json`
- Results: `src/results/raw/`
- Prompts: `src/prompts/`

---

## 📝 Data Format

### Input: Interactions
```json
{
  "thread_id": {
    "id": 123,
    "text": "Hello assistant",
    "role": 0,  // 0=teacher/user, 1=chatbot/assistant
    "created": "2026-04-14 12:00:00"
  }
}
```

### Input: Ground Truth Labels
```json
{
  "thread_id": {
    "message_id": 123,
    "response_stance": "confident",
    "prompt_type": "solution_request",
    "interactional_move": "probe_more_info",
    ...
  }
}
```

### Output: Results
```json
{
  "thread_id": "...",
  "message_id": 123,
  "text": "Hello assistant...",
  "category": "response_stance",
  "strategy": "few_shot",
  "model": "gpt4o",
  "role": 1,
  "true_label": "confident",
  "predicted_label": "confident",
  "match": true
}
```

---

## 📊 Sample Output

```
Loading data...
   ✓ Loaded 409 threads
   ✓ Loaded 3600 messages
   ✓ Loaded ground truth labels

======================================================================
  STARTING EXPERIMENT
  2026-04-14 12:34:56
  Models: gpt4o
  Categories: response_substance
======================================================================

  [1/1] Model: gpt4o

  [5/6] response_substance (role=chatbot)
    [1/2] basic
      ✓ msg 2390: elaborate ✓
      → 5 predictions
    [2/2] few_shot
      ✓ msg 2392: elaborate ✓
      → 5 predictions

======================================================================
  SAVING RESULTS
======================================================================

  Saved 10 results
      src\results\raw\results_20260414_123456.json

======================================================================
STATISTICS
======================================================================

  Total predictions:     10
  ✓ Successful:          8
  ✗ Failed:              2
  ⊘ Skipped:             0
  Overall accuracy:      80.00%

  ──────────────────────────────────────────────────────────────────
  PER-MODEL ACCURACY
  ──────────────────────────────────────────────────────────────────

  gpt4o           80.00% (8/10)

  ──────────────────────────────────────────────────────────────────
  PER-CATEGORY ACCURACY
  ──────────────────────────────────────────────────────────────────

  response_substance     80.00% (8/10)
```

## Logging & Debugging

**Console Output:**
- Real-time progress during experiment
- ERROR and above level messages

**File Logs:**
- Location: `logs/experiment_TIMESTAMP.log`
- Includes DEBUG level information
- Full stack traces for errors
- Timestamped for easy tracking

Example: `logs/experiment_20260414_123456.log`

---

## ⚙️ Dependencies

- Python 3.10+
- OpenAI SDK (GPT-4o access)
- Anthropic SDK (Claude access)
- Ollama (LLaMA 3, local)
- pandas (for metrics)
- scikit-learn (for evaluation metrics)

---

## 📄 License

Bachelor thesis project in Web Programing - 2026

---

## 👤 Author

Ester and Leia

---

**Last Updated:** April 14, 2026

*For testing (smaller dataset):*
--> interactions data/process_data/test_interactions.json
--> ground-truth data/process_data/processed_ground_truths.json


