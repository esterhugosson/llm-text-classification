# Error handling and exception management

import functools
import traceback
from typing import Callable, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentError(Exception):
    """Base exception for all experiment-related errors"""
    pass


class DataLoadError(ExperimentError):
    """Raised when data loading fails"""
    pass


class ClassificationError(ExperimentError):
    """Raised when classification fails"""
    pass


class ConfigError(ExperimentError):
    """Raised when configuration is invalid"""
    pass


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors and log them
    
    Usage:
        @handle_errors
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except ExperimentError as e:
            logger.error(f"Experiment error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ExperimentError(f"Error in {func.__name__}: {str(e)}") from e
    
    return wrapper


def log_exception(exc: Exception, context: str = "") -> None:
    """
    Log an exception with context
    
    Args:
        exc: Exception to log
        context: Additional context about where error occurred
    """
    logger.error(f"Exception occurred {context}: {type(exc).__name__}: {str(exc)}")
    logger.debug(traceback.format_exc())


def validate_or_error(condition: bool, error_class: type, message: str) -> None:
    """
    Validate condition and raise error if false
    
    Usage:
        validate_or_error(len(data) > 0, DataLoadError, "No data loaded")
    """
    if not condition:
        logger.error(f"Validation failed: {message}")
        raise error_class(message)
