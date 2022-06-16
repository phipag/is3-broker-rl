import logging.config
import os
import sys

from is3_broker_rl.utils import get_root_path


def setup_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "precise": {
                    "format": "[%(asctime)s:%(name)s:%(levelname)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "brief": {"format": "[%(name)s:%(levelname)s] %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "precise",
                    "filename": get_root_path() / os.getenv("LOG_FILE", "logs/main.log"),
                    "mode": "a",
                },
            },
            "loggers": {},
            "root": {"level": os.getenv("LOG_LEVEL", logging.INFO), "handlers": ["console", "file"]},
        }
    )
