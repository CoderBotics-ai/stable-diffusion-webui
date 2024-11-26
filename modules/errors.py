"""
Error handling and reporting module for the Stable Diffusion Web UI.
Provides functionality for tracking, formatting and displaying exceptions,
as well as version compatibility checking.
"""

import sys
import textwrap
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from packaging import version
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Constants
MAX_EXCEPTION_RECORDS = 5
ERROR_PREFIX = "***"
SEPARATOR = "="

@dataclass
class ExceptionRecord:
    """Structured format for storing exception information."""
    exception: str
    traceback: List[List[str]]

class ExceptionTracker:
    """Manages tracking and storage of exception records."""
    
    def __init__(self):
        self._records: List[ExceptionRecord] = []
        self._displayed_errors: Dict[str, bool] = {}

    def format_traceback(self, tb) -> List[List[str]]:
        """Format a traceback into a structured list of filename, line number, and code."""
        return [[f"{x.filename}, line {x.lineno}, {x.name}", x.line] 
                for x in traceback.extract_tb(tb)]

    def format_exception(self, e: Exception, tb) -> ExceptionRecord:
        """Convert an exception and its traceback into a structured record."""
        return ExceptionRecord(
            exception=str(e),
            traceback=self.format_traceback(tb)
        )

    def record_exception(self) -> None:
        """Record the current exception information if available."""
        _, e, tb = sys.exc_info()
        if e is None:
            return

        if self._records and str(self._records[-1].exception) == str(e):
            return

        self._records.append(self.format_exception(e, tb))
        if len(self._records) > MAX_EXCEPTION_RECORDS:
            self._records.pop(0)

    def get_exceptions(self) -> Union[List[ExceptionRecord], str]:
        """Retrieve the list of recorded exceptions."""
        try:
            return list(reversed(self._records))
        except Exception as e:
            return str(e)

# Global exception tracker instance
exception_tracker = ExceptionTracker()

def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    
    Args:
        message: The error message to display
        exc_info: Whether to include the full exception traceback
    """
    exception_tracker.record_exception()
    
    for line in message.splitlines():
        logger.error(f"{ERROR_PREFIX} {line}")
    
    if exc_info:
        logger.error(textwrap.indent(traceback.format_exc(), "    "))
        logger.error("---")

def print_error_explanation(message: str) -> None:
    """
    Format and print a detailed error explanation message.
    
    Args:
        message: The explanation message to display
    """
    exception_tracker.record_exception()
    lines = message.strip().split("\n")
    max_len = max(len(x) for x in lines)

    logger.error(SEPARATOR * max_len)
    for line in lines:
        logger.error(line)
    logger.error(SEPARATOR * max_len)

def display(e: Exception, task: Optional[str], *, full_traceback: bool = False) -> None:
    """
    Display an exception with optional task context and full traceback.
    
    Args:
        e: The exception to display
        task: Optional task description where the error occurred
        full_traceback: Whether to include the full traceback including frames before the try-catch
    """
    exception_tracker.record_exception()
    
    logger.error(f"{task or 'error'}: {type(e).__name__}")
    te = traceback.TracebackException.from_exception(e)
    
    if full_traceback:
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    
    logger.error(''.join(te.format()))

    if "copying a param with shape torch.Size([640, 1024]) from checkpoint" in str(e):
        print_error_explanation("""
        The most likely cause of this is you are trying to load Stable Diffusion 2.0 model
        without specifying its config file. See:
        https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20
        """)

def display_once(e: Exception, task: str) -> None:
    """Display an exception only once for a given task."""
    exception_tracker.record_exception()
    
    if task not in exception_tracker._displayed_errors:
        display(e, task)
        exception_tracker._displayed_errors[task] = True

def run(code: callable, task: str) -> None:
    """
    Execute code with exception handling and task context.
    
    Args:
        code: Callable to execute
        task: Description of the task being performed
    """
    try:
        code()
    except Exception as e:
        display(e, task)

class VersionChecker:
    """Handles version compatibility checking for dependencies."""
    
    EXPECTED_VERSIONS = {
        'torch': "2.1.2",
        'xformers': "0.0.23.post1",
        'gradio': "3.41.2"
    }

    @classmethod
    def check_versions(cls) -> None:
        """Check versions of critical dependencies against expected versions."""
        from modules import shared
        import torch
        import gradio

        if version.parse(torch.__version__) < version.parse(cls.EXPECTED_VERSIONS['torch']):
            cls._show_torch_version_error(torch.__version__)

        if shared.xformers_available:
            import xformers
            if version.parse(xformers.__version__) < version.parse(cls.EXPECTED_VERSIONS['xformers']):
                cls._show_xformers_version_error(xformers.__version__)

        if gradio.__version__ != cls.EXPECTED_VERSIONS['gradio']:
            cls._show_gradio_version_error(gradio.__version__)

    @classmethod
    def _show_torch_version_error(cls, current_version: str) -> None:
        print_error_explanation(f"""
        You are running torch {current_version}.
        The program is tested to work with torch {cls.EXPECTED_VERSIONS['torch']}.
        To reinstall the desired version, run with commandline flag --reinstall-torch.
        Beware that this will cause a lot of large files to be downloaded, as well as
        there are reports of issues with training tab on the latest version.

        Use --skip-version-check commandline argument to disable this check.
        """.strip())

    @classmethod
    def _show_xformers_version_error(cls, current_version: str) -> None:
        print_error_explanation(f"""
        You are running xformers {current_version}.
        The program is tested to work with xformers {cls.EXPECTED_VERSIONS['xformers']}.
        To reinstall the desired version, run with commandline flag --reinstall-xformers.

        Use --skip-version-check commandline argument to disable this check.
        """.strip())

    @classmethod
    def _show_gradio_version_error(cls, current_version: str) -> None:
        print_error_explanation(f"""
        You are running gradio {current_version}.
        The program is designed to work with gradio {cls.EXPECTED_VERSIONS['gradio']}.
        Using a different version of gradio is extremely likely to break the program.

        Reasons why you have the mismatched gradio version can be:
        - you use --skip-install flag
        - you use webui.py to start the program instead of launch.py
        - an extension installs the incompatible gradio version

        Use --skip-version-check commandline argument to disable this check.
        """.strip())

# Alias the version checker's check_versions method for backwards compatibility
check_versions = VersionChecker.check_versions