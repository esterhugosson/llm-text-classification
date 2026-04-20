import traceback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentError(Exception):
    """For all experiment-related errors"""
    pass


class DataLoadError(ExperimentError):
    """Given when data loading fails"""
    pass


class ClassificationError(ExperimentError):
    """Given when classification fails"""
    pass


class ConfigError(ExperimentError):
    """Given when configuration is invalid"""
    pass


def log_exception(exc: Exception, context: str = "") -> None:
    logger.error(f"Exception occurred {context}: {type(exc).__name__}: {str(exc)}")
    logger.debug(traceback.format_exc())


def validate_or_error(condition: bool, error_class: type, message: str) -> None:
    if not condition:
        logger.error(f"Validation failed: {message}")
        raise error_class(message)