
"""Singleton-related helpers and metaclass implementations."""

# http://norvig.com/python-iaq.html
def singleton(object):
    """
    Raise an exception if an object of this class has been instantiated before.
    :param object:
    :returns:
    :raises ValueError:
    """
    cls = object.__class__
    if hasattr(cls, '__instantiated'):
        raise ValueError(f"{cls} is a Singleton class but is already instantiated")
    cls.__instantiated = True


class Singleton(type):
    """
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    Metaclass for singletons. Any instantiation of a Singleton class yields
    exact same object, e.g.:
    >>> class MyClass(metaclass=Singleton):
    >>>    pass
    >>> a = MyClass()
    >>> b = MyClass()
    >>> a is b
    >>> True

    Note: initialization arguments provided after the first instantiation
    are ignored, since the already created instance is returned.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
