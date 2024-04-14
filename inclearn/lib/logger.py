from loguru import logger
import sys


def set_logging_level(logging_level):
    logging_level = logging_level.lower()
    logger.remove()

    if logging_level == "critical":
        level = "CRITICAL"
    elif logging_level == "warning":
        level = "WARNING"
    elif logging_level == "info":
        level = "INFO"
    else:
        level = "DEBUG"

    logger.add(sys.stderr, level=level)
