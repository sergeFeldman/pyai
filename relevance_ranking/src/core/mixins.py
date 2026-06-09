from logging import Logger
from typing import Optional


class BaseLoggable:
    """
    A mixin class that provides a standardized, class-level loggable interface.

    Any class, which inherits from BaseLoggable will gain access to
    a 'logger()' class method, which provides a pre-configured logger
    instance names after the class itself.
    This ensures consistent logging behavior across the application.
    """
    _logger: Optional[Logger] = None

    @classmethod
    def logger(cls) -> Logger:
        """
        Accessor for the class-level _logger attribute.

        If a logger already exists for this class or any of its parents,
        it is returned. Otherwise, a new Logger instance is created,
        cached, and returned.
        """
        # Walk the Method Resolution Order (MRO) to find an inherited logger.
        # This allows a child class to use a logger set by a parent.
        for class_obj in cls.__mro__:
            if hasattr(class_obj, '_logger') and getattr(class_obj, '_logger') is not None:
                return class_obj._logger

        # If no logger is found, create one for this specific class.
        cls._logger = Logger(name=cls.__name__)
        return cls._logger

    @classmethod
    def set_logger(cls, logger: Logger) -> None:
        """
        Allows for explicitly setting a logger instance for this class.

        This is useful for dependency injection during testing or for configuring
        a parent logger for a group of related classes.

        Args:
            logger (Logger): Provided logger instance to use.
        """
        cls._logger = logger
