import json
from typing import Dict, List, Any
from src.utils.error_handler import DataLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_interactions(path: str) -> Dict[str, List[Dict[str, Any]]]:
    
    if not path:
        msg = "Interactions path not provided"
        logger.error(msg)
        raise DataLoadError(msg)
    
    logger.debug(f"Loading interactions from: {path}")
    
    # Load JSON and send error if file is not found or valid
    try:
        with open(path, "r", encoding="utf-8") as f:
            interactions_data = json.load(f)
    except FileNotFoundError:
        msg = f"Interactions file not found: {path}"
        logger.error(msg)
        raise DataLoadError(msg) from None
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in interactions file {path}: {str(e)}"
        logger.error(msg)
        raise DataLoadError(msg) from e
    except Exception as e:
        msg = f"Failed to load interactions from {path}: {str(e)}"
        logger.error(msg)
        raise DataLoadError(msg) from e
    
    # Check if structure is correct (dict of thread_id to list of interactions)
    if not isinstance(interactions_data, dict):
        msg = f"Invalid interactions format: expected dict, got {type(interactions_data).__name__}"
        logger.error(msg)
        raise DataLoadError(msg)
    
    # Check if each thread has list of interactions with required fields
    for thread_id, messages in interactions_data.items():
        if not isinstance(messages, list):
            msg = f"Invalid thread {thread_id}: messages should be list, got {type(messages).__name__}"
            logger.error(msg)
            raise DataLoadError(msg)
        
        for i, msg_dict in enumerate(messages):
            required_fields = ['id', 'text', 'created', 'role']
            missing = [f for f in required_fields if f not in msg_dict]
            if missing:
                msg = f"Thread {thread_id}, message {i}: missing fields {missing}"
                logger.error(msg)
                raise DataLoadError(msg)
    
    logger.info(f"[DONE] Loaded interactions: {len(interactions_data)} threads")
    return interactions_data

