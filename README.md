# LLM Text Classification

**An experimental evaluation of prompt-based LLM classification against human qualitative coding**

This project was made for our bachelor thesis in computer science and compares three LLMs (GPT-4o, Claude Sonnet, LLaMA 3) in classifying chatbot interactions across 6 dialogue act categories. It evaluates the models' performance against human qualitative coding and analyzes the impact of different prompting strategies. The current version contains conversational context in the prompts.

---

## Quick Start

### Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run with custom options
python -m src --limit 5 --categories is_followup --models gpt4o --strategy few_shot
```

### Output

Results are saved to `src/results/raw/results_TIMESTAMP.json`
Logs are saved to `logs/experiment_TIMESTAMP.log`

---

## Project Structure

```
src/
├── experiments/              # Main experiment code
├── pipeline/                 # Classification pipeline
├── llm/                      # LLM models (GPT-4o, Claude, LLaMA 3)
├── data/                     # Data loading
├── evaluation/               # Metrics and analysis
├── prompts/                  # Prompt templates
└── results/                  # Results and logs
```

---

### Available Options

| Flag | Example | Description |
|------|---------|-------------|
| `--limit` | `--limit 10` | Max messages per category |
| `--models` | `--models gpt4o claude` | Which models to test |
| `--categories` | `--categories response_stance` | Which categories to test |
| `--strategy` | `--strategy basic` | Prompting strategy (basic, few_shot) |

---

## Categories

### Dialogue Act Categories (6 total)

| Category | Description |
|----------|-------------|
| `response_stance` | How chatbot responds (none/confident/neutral) |
| `prompt_type` | Type of teacher prompt |
| `interactional_move` | Chatbot's interactive move |
| `cps_behavior` | Teacher's problem-solving behavior |
| `response_substance` | Substance of chatbot's response |
| `is_followup` | Whether message is a follow-up |

---

## Dependencies

- Python 3.10+
- OpenAI SDK
- Anthropic SDK
- Ollama
- pandas
- scikit-learn

---

## Authors

Ester and Leia
