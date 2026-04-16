import json
from typing import Dict, List, Any
from src.utils.error_handler import DataLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_ground_truths(path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not path:
        msg = "Ground truth path not provided"
        logger.error(msg)
        raise DataLoadError(msg)
    
    logger.debug(f"Loading ground truth from: {path}")
    
    # Load JSON and send error if file is not found or valid
    try:
        with open(path, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)
    except FileNotFoundError:
        msg = f"Ground truth file not found: {path}"
        logger.error(msg)
        raise DataLoadError(msg) from None
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in ground truth file {path}: {str(e)}"
        logger.error(msg)
        raise DataLoadError(msg) from e
    except Exception as e:
        msg = f"Failed to load ground truth from {path}: {str(e)}"
        logger.error(msg)
        raise DataLoadError(msg) from e
    
    # cHEck if structure is correct (dict of thread_id to list of labels)
    if not isinstance(ground_truths, dict):
        msg = f"Invalid ground truth format: expected dict, got {type(ground_truths).__name__}"
        logger.error(msg)
        raise DataLoadError(msg)
    
    # check if each thread has list of labels with required fields
    for thread_id, labels_list in ground_truths.items():
        if not isinstance(labels_list, list):
            msg = f"Invalid thread {thread_id}: labels should be list, got {type(labels_list).__name__}"
            logger.error(msg)
            raise DataLoadError(msg)
        
        for i, label_dict in enumerate(labels_list):
            if not isinstance(label_dict, dict):
                msg = f"Thread {thread_id}, label {i}: expected dict, got {type(label_dict).__name__}"
                logger.error(msg)
                raise DataLoadError(msg)
    
    logger.info(f"[DONE] Loaded ground truth: {len(ground_truths)} threads")
    return ground_truths