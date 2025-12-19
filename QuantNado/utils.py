import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_path: Path, verbose: bool):
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> [<magenta>{level}</magenta>] <level>{message}</level>"
    logger.add(log_path, level="DEBUG", format=log_format, mode="w", colorize=False)
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format=log_format,
        colorize=True,
    )
