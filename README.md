# llm-text-classification
Bachelor thesis for exploring how to classify educational chatbot interactions using LLMs, with set labels using multi-class classifications.

## Start
    ` python -m src `

## Dependencies

## How to get started:

## Data

## Categories and possible Labels

## Project structure
```
src/
│
├── evaluation/
│   ├── 
│   ├── 
│   ├── 
│   └── 
│
├── experiments/
│   ├── runner.py              # Orkestrerar hela experimentet
│   ├── config.py              # MODELS, CATEGORIES, STRATEGIES, PATHS
│   ├── experiment.py          # Experiment-klass (kärnlogik)
│   └── stats.py               # Statistikhantering
│
├── data/
│   ├── loaders/
│   │   ├── interaction_loader.py
│   │   ├── ground_truth_loader.py
│   │   └── batch_iterator.py
│   │
│   └── models/
│       └── data_models.py     # dataclasses (Message, PredictionResult)
│
├── pipeline/
│   ├── filter.py              # filtrering (role, text length etc)
│   ├── matcher.py             # matchning mot ground truth
│   ├── classifier_pipeline.py # kör klassificering
│   └── result_builder.py      # bygger result rows
│
├── llm/
│   ├── claude_sonnet.py       
│   ├── gpt_4o.py           
│   ├── llama_3.py 
│   └── prompt_loader.py     
│
├── prompts/
│   ├── cps_behaviour/
│   │   ├── basic.txt
│   │   └── few_shot.txt
│   ├── interactional_move/
│   │   ├── basic.txt
│   │   └── few_shot.txt
│   ├── is_followup/
│   │   ├── basic.txt
│   │   └── few_shot.txt
│   ├── prompt_type/
│   │   ├── basic.txt
│   │   └── few_shot.txt
│   ├── response_stance/
│   │   ├── basic.txt
│   │   └── few_shot.txt
│   └── response_substance/
│       ├── basic.txt
│       └── few_shot.txt
│
├── ui/
│   ├── cli.py                
│   └── tui.py                 
│
├── utils/
│   ├── logger.py
│   └── file_utils.py
│
└── results/
    └── writer.py              # sparar JSON
```

## Project structure (nuvarande structure)
```
src/
├── evaluation/
│   ├── run_results.py    # Mini-test med metrics (kommer denna ens användas?)
│   └── metrics.py        # Evaluering av resultat
│
├── experiments/
│   ├── runner.py         # Entry point
│   ├── config.py         # MODELS, CATEGORIES, STRATEGIES
│   ├── experiment.py     # Experiment logic
│   └── stats.py          # Statistics
│
├── data/
│   ├── loaders/
│   │   ├── interaction_loader.py
│   │   ├── ground_truth_loader.py
│   │   └── batch_iterator.py (inte implemententerad)
│   └── models/
│       └── data_models.py
│
├── pipeline/
│   ├── filter.py
│   ├── matcher.py
│   ├── classifier_pipeline.py
│   └── result_builder.py
│
├── llm/
│   ├── claude_sonnet.py
│   ├── gpt_4o.py
│   ├── llama_3.py
│   └── prompt_loader.py
│
├── prompts/
│   └── [6 categories with basic.txt & few_shot.txt]
│
└── results/
    └── raw/  # Sparade resultat
```
## Models

## How to run code

### Full experiment (right now I don't think this will work, but is meant to do all interactions available)
`python -m src.experiments.runner`

### Filtering options example
`python -m src.experiments.runner --limit 4 --categories interactional_move --models gpt4o`

## Recent Changes ->
Refactored the experiment pipeline to support:

- Message limiting: --limit N to test with only N messages per category
- Selective model testing: --models gpt4o|claude|llama3 to test specific LLMs
- Selective category testing: --categories <name> to test specific dialogue acts
- Test mode: --test flag for quick validation (GPT-4 only, 2 messages)

**Code changes made:**
Added message_limit parameter to Experiment.__init__()
Added limit check in run() method loop (breaks after N messages per category)
Updated runner.py to pass message_limit from CLI arguments

(Having some problems with paths so have to doublecheck right paths in all files)

*Default paths (in runner.py, lines ~54-55):*
--> interactions data/process_data/processed_interactions.json
--> ground-truth data/process_data/processed_ground_truths.json

*For testing (smaller dataset):*
--> interactions data/process_data/test_interactions.json
--> ground-truth data/process_data/processed_ground_truths.json


