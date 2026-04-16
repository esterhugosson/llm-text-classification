import os
from pathlib import Path
from src.utils.error_handler import validate_or_error, ConfigError

MODELS = {
    "gpt4o": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "llama3": "llama3:8b"
}

CATEGORIES = {
    # "response_stance": 1,      # Only chatbot responses (role=1)
    "prompt_type": 0,          # Only teacher prompts (role=0)
    "interactional_move": 1,   # Only chatbot moves (role=1)
    "cps_behavior": 0,         # Only teacher prompts (role=0)
    #"response_substance": 1,   # Only chatbot responses (role=1)
    "is_followup": None,       # All speakers
}

STRATEGIES = ["basic", "few_shot"]

TEMPERATURE = 0
TOP_K = 1
MAX_TOKENS = 20

# Base data directory
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESS_DATA_DIR = DATA_DIR / "process_data"

# Input file paths
INTERACTIONS_PATH = str(PROCESS_DATA_DIR / "processed_interactions_without_examples.json")
GROUND_TRUTH_PATH = str(PROCESS_DATA_DIR / "processed_ground_truths_plus.json")

# Output directory
RESULTS_DIR = Path("src/results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prompts directory
PROMPTS_DIR = Path("src/prompts")



def validate_config() -> bool:
    
    validate_or_error(
        Path(INTERACTIONS_PATH).exists(),
        ConfigError,
        f"Interactions file not found: {INTERACTIONS_PATH}"
    )
    
    validate_or_error(
        Path(GROUND_TRUTH_PATH).exists(),
        ConfigError,
        f"Ground truth file not found: {GROUND_TRUTH_PATH}"
    )
    
    validate_or_error(
        PROMPTS_DIR.exists(),
        ConfigError,
        f"Prompts directory not found: {PROMPTS_DIR}"
    )

    for category in CATEGORIES.keys():
        for strategy in STRATEGIES:
            prompt_path = PROMPTS_DIR / category / f"{strategy}.txt"
            validate_or_error(
                prompt_path.exists(),
                ConfigError,
                f"Prompt not found: {prompt_path}"
            )
    
    return True