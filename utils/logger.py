import logging
import torch.distributed as dist
from typing import Optional, Union


logger_initialized: dict[str, bool] = {}

def get_logger(name: str, log_file: Optional[str]=None, log_level: int=logging.INFO, file_mode: str='w') -> logging.Logger:
    """
    Initialize and get a logger with the given name.

    If a logger with the specified name has already been initialized,
    it returns the existing logger. Otherwise, it creates a new one.
    
    Args:
        name (str): Name of the logger.
        log_file (Optional[str]): Path to the log file. If None, no file logging is done.
        log_level (int): Logging level.
        file_mode (str): File mode for the log file ('w' for write, 'a' for append, etc.).

    Returns:
        logging.Logger: The initialized logger object.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    
    # Prevents reinitializing already initialized loggers, especially in hierarchical logger structures.
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    
    # Adjust log level of existing StreamHandlers to avoid duplicate logging in certain environments.
    for handler in logger.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # Determine the rank in a distributed setting; defaults to 0 if not available or initialized.
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    # File logging is only enabled for rank 0 in a distributed setting.
    if rank == 0 and log_file:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Assign formatter and log level to each handler and attach them to the logger.
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level if rank == 0 else logging.error)
    logger_initialized[name] = True

    return logger


def get_root_logger(log_file: Optional[str]=None, log_level: int=logging.INFO, name: str='main') -> logging.Logger:
    """
    Get a root logger with an optional file handler.

    This function is specifically for initializing the main logger of an application.

    Args:
        log_file (Optional[str]): Path to the log file. If None, no file logging is done.
        log_level (int): Logging level.
        name (str): Name of the logger, defaults to 'main'.

    Returns:
        logging.Logger: The initialized root logger object.
    """
    logger = get_logger(name, log_file, log_level)
    logging_filter = logging.Filter(name)
    logger.addFilter(logging_filter)

    return logger

def print_log(msg: str, logger: Union[None, str]=None, level: int=logging.INFO):
    """
    Print a log message either to the console, to a specified logger, or silently.

    This function provides a flexible way of logging messages. It can print to the console, use a given logger,
    or silently ignore the message based on the `logger` argument.

    Args:
        msg (str): The message to log.
        logger (Union[None, str]): Specifies the logging behavior. It can be:
            - None: prints the message to the console.
            - A logging.Logger object: logs the message using this logger.
            - 'silent': does nothing (silently ignores the message).
            - A string: assumes it's the name of a logger to fetch and use for logging.
        level (int): The logging level for the message (e.g., logging.INFO, logging.ERROR, etc.).

    Raises:
        TypeError: If the `logger` argument is not one of the expected types.
    """

    if logger is None:
        # Print the message to the console if no logger is specified.
        print(msg)
    elif isinstance(logger, logging.Logger):
        # If a Logger object is provided, use it to log the message.
        logger.log(level=level, msg=msg)
    elif logger == 'silent':
        # If 'silent' is specified, do nothing (i.e., silently ignore the message).
        pass
    elif isinstance(logger, str):
        # If a string is provided, treat it as the name of a logger to fetch and use.
        _logger = get_logger(logger)
        _logger.log(level=level, msg=msg)
    else:
        # Raise an error if the `logger` argument is not one of the expected types.
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
