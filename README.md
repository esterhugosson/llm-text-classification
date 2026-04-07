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

## Models

