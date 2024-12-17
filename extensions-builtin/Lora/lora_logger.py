import sys
from typing import Final, Dict
import copy
import logging


class ColoredFormatter(logging.Formatter):
    """A custom formatter that adds ANSI color codes to log level names.
    
    This formatter wraps the log level name with appropriate ANSI color codes
    to provide colored output in terminals that support it.
    """
    
    COLORS: Final[Dict[str, str]] = {
        "DEBUG": "\033[0;36m",     # CYAN
        "INFO": "\033[0;32m",      # GREEN
        "WARNING": "\033[0;33m",   # YELLOW
        "ERROR": "\033[0;31m",     # RED
        "CRITICAL": "\033[0;37;41m", # WHITE ON RED
        "RESET": "\033[0m",        # RESET COLOR
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colored level name.

        Args:
            record: The log record to format.

        Returns:
            str: The formatted log message with colored level name.
        """
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        color_seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{color_seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Initialize logger
logger = logging.getLogger("lora")
logger.propagate = False

# Add handler if none exists
if not logger.handlers:
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter("[%(name)s]-%(levelname)s: %(message)s")
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)