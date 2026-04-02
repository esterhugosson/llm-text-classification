MODELS = {
    "gpt4o": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "llama3": "llama3:8b"
}

CATEGORIES = {
    "response_stance": 1,      # Only chatbot responses (role=1)
    "prompt_type": 0,          # Only teacher prompts (role=0)
    "interactional_move": 1,   # Only chatbot moves (role=1)
    "cps_behavior": 0,         # Only teacher prompts (role=0)
    "response_substance": 1,   # Only chatbot responses (role=1)
    "is_followup": None,       # All speakers
}

STRATEGIES = ["basic", "few_shot"]

TEMPERATURE = 0