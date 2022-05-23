from pathlib import Path

import is3_broker_rl


def get_root_path() -> Path:
    """
    This is the path to the root directory containing the pyproject.toml file.
    :return: Path to the root directory
    """
    return Path(is3_broker_rl.__file__).parent.parent
