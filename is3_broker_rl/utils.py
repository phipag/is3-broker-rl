from pathlib import Path

import is3_broker_rl


def get_root_path() -> Path:
    return Path(is3_broker_rl.__file__).parent.parent
