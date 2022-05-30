import logging

from ray import serve
from starlette.requests import Request


@serve.deployment(route_prefix="/observe")
class ObservationController:
    def __init__(self):
        self._log = logging.getLogger(__name__)

    def __call__(self, request: Request):
        self._log.debug(request)


# Uncomment to disable API endpoint
ObservationController.deploy()
