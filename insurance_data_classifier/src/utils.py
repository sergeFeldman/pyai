from datetime import datetime
import logging
import os
import sys


def config_logging(module_name: str,
                   log_level: int = logging.INFO,
                   console_output: bool = True) -> logging.Logger:
    """
    Setup logging configuration.
    """
    if module_name == "__main__":
        display_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    else:
        display_name = module_name.split('.')[-1]

    logger = logging.getLogger(display_name)
    logger.setLevel(log_level)

    # Clear any existing handlers to avoid duplicates.
    logger.handlers.clear()

    # Create formatter.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f"logs/{display_name}_{get_timestamp()}.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_timestamp() -> str:
    """
    Get current timestamp for file naming.

    Returns:
        String timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
