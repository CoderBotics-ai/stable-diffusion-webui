import os
from modules import paths

def preload(parser):
    """
    Preloads the command line argument for specifying the path to LDSR model files.

    Args:
        parser: The argument parser instance to which the argument will be added.
    """
    parser.add_argument(
        "--ldsr-models-path",
        type=str,
        help="Path to directory with LDSR model file(s).",
        default=os.path.join(paths.models_path, 'LDSR')
    )