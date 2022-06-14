import logging

import dotenv
import ray
from ray import serve

from conf import setup_logging


def main():
    setup_logging()
    dotenv.load_dotenv(override=False)
    log = logging.getLogger(__name__)
    log.info("Starting Ray server ...")
    ray.init()
    serve.start()


if __name__ == "__main__":
    main()
