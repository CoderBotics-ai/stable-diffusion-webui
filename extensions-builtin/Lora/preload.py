import os
from os import PathLike
from typing import Any
from modules import paths
from modules.paths_internal import normalized_filepath


def preload(parser: Any) -> None:
    """
    Adds command-line arguments for Lora and LyCORIS network directories to the argument parser.

    Args:
        parser: The argument parser to add the arguments to.
    """
    parser.add_argument(
        "--lora-dir",
        type=normalized_filepath,
        help="Path to directory with Lora networks.",
        default=os.path.join(paths.models_path, 'Lora')
    )
    parser.add_argument(
        "--lyco-dir-backcompat",
        type=normalized_filepath,
        help="Path to directory with LyCORIS networks (for backawards compatibility; can also use --lyco-dir).",
        default=os.path.join(paths.models_path, 'LyCORIS')
    )