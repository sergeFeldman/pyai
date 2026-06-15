"""Singleton-related helpers and metaclass implementations."""


def singleton(obj):
    """Raise ValueError if an instance of this class has already been created.

    Function-based alternative to the Singleton metaclass for cases where only
    one instance should ever exist and subsequent instantiation is an error
    rather than silently returning the first instance.

    Args:
        obj: Newly constructed instance to check.

    Raises:
        ValueError: If the class has been instantiated before.
    """
    cls = obj.__class__
    if hasattr(cls, '__instantiated'):
        raise ValueError(f"{cls} is a Singleton class but is already instantiated")
    cls.__instantiated = True


class Singleton(type):
    """Metaclass that ensures only one instance of a class exists per process.

    Any instantiation after the first returns the already-created instance:

        class MyClass(metaclass=Singleton):
            pass

        a = MyClass()
        b = MyClass()
        a is b  # True

    Note: initialization arguments provided after the first instantiation
    are ignored, since the already-created instance is returned.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
