import logging
import sys

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(verbose: bool = False, log_file: str = "verse.log") -> None:
    """
    Configures centralized logging for the entire package.

    Args:
        verbose (bool): If True, sets logging level to DEBUG. Otherwise, INFO.
        log_file (str): Path to the log file.
    """
    root_package = __name__.split(".")[0]
    logging_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger(root_package)
    logger.setLevel(logging_level)
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
