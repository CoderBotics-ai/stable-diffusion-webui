import sys
from copy import copy
import logging
from typing import Any


class ColoredFormatter(logging.Formatter):
    """A custom formatter that adds color to log level names.

    This formatter wraps log level names with ANSI color codes to provide
    colored output in terminals that support it.
    """

    COLORS: dict[str, str] = {
        "DEBUG": "\033[0;36m",     # CYAN
        "INFO": "\033[0;32m",      # GREEN
        "WARNING": "\033[0;33m",   # YELLOW
        "ERROR": "\033[0;31m",     # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",        # RESET COLOR
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colored level name.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message with colored level name.
        """
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Initialize logger
logger = logging.getLogger("lora")
logger.propagate = False

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("[%(name)s]-%(levelname)s: %(message)s")
    )
    logger.addHandler(handler)