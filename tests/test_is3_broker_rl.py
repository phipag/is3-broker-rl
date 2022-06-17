from is3_broker_rl import __version__
from is3_broker_rl.conf.log import setup_logging


def test_version():
    assert __version__ == "0.1.0"


def test_setup_logging_ok():
    # Should not throw anything
    setup_logging()
