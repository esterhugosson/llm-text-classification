# Configuration for LLM Classification Experiment

import os
from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS = {
    "gpt4o": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "llama3": "llama3:8b"
}


# ============================================================================
# CLASSIFICATION CATEGORIES
# ============================================================================

CATEGORIES = {
    "response_stance": 1,      # Only chatbot responses (role=1)
    "prompt_type": 0,          # Only teacher prompts (role=0)
    "interactional_move": 1,   # Only chatbot moves (role=1)
    "cps_behavior": 0,         # Only teacher prompts (role=0)
    "response_substance": 1,   # Only chatbot responses (role=1)
    "is_followup": None,       # All speakers
}


# ============================================================================
# STRATEGIES
# ============================================================================

STRATEGIES = ["basic", "few_shot"]


# ============================================================================
# LLM PARAMETERS
# ============================================================================

TEMPERATURE = 0
TOP_K = 0.9
MAX_TOKENS=200

# ============================================================================
# DATA PATHS (with environment variable support)
# ============================================================================

# Base data directory
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESS_DATA_DIR = DATA_DIR / "process_data"

# Input files
INTERACTIONS_PATH = str(PROCESS_DATA_DIR / "processed_interactions.json")
GROUND_TRUTH_PATH = str(PROCESS_DATA_DIR / "processed_ground_truths.json")

# Output directory
RESULTS_DIR = Path("src/results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prompts directory
PROMPTS_DIR = Path("src/prompts")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> bool:
    """Validate that all required files exist"""
    from src.utils.error_handler import validate_or_error, ConfigError
    
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
    
    return True