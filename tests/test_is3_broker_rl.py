import is3_broker_rl
from is3_broker_rl import __version__
from is3_broker_rl.conf.log import setup_logging
from is3_broker_rl.utils import get_root_path


def test_version():
    assert __version__ == "0.1.0"


def test_setup_logging_ok():
    # Should not throw anything
    setup_logging()


def test_get_root_path_ok():
    root_path = get_root_path()
    # The root folder should contain a folder named like the main package
    assert any(str(path).find(is3_broker_rl.__name__) != -1 for path in root_path.iterdir())
