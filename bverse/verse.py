import logging

from bverse.utils.logger import configure_logging
from bverse.config import Config
from bverse.orchestrator import Orchestrator

configure_logging(verbose=True)
logger = logging.getLogger(__name__)


def verse(config: Config | None = None) -> None:
    """
    Launches the Verse real-time conversation assistant with the provided configuration.

    Args:
        config: Optional instance of Config specifying runtime parameters.
            If not provided, the default Config instance is used.
    """
    Orchestrator(config or Config()).run()


if __name__ == "__main__":
    verse()
