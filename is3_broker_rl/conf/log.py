import logging.config
import os
import sys
from pathlib import Path

import is3_broker_rl


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
                    "filename": Path(is3_broker_rl.__file__).parent.parent / os.getenv("LOG_FILE", "logs/main.log"),
                    "mode": "w",
                },
            },
            "loggers": {},
            "root": {"level": os.getenv("LOG_LEVEL", logging.INFO), "handlers": ["console", "file"]},
        }
    )
