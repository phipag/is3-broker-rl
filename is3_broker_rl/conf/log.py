import logging.config
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_filename: Optional[str] = None) -> None:
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
                    "filename": Path(os.getenv("LOG_DIR", "logs/")) / (log_filename or "main.log"),
                    "mode": "a+",
                },
            },
            "loggers": {},
            "root": {"level": os.getenv("LOG_LEVEL", logging.INFO), "handlers": ["console", "file"]},
        }
    )
