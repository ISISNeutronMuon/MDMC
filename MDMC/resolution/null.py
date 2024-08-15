"""A class for the null object for Resolution classes."""
from typing import Any

from MDMC.resolution.resolution import Resolution


class NullResolution(Resolution):
    """
    The null object for the Resolution class.
    Used when there is no resolution to apply.
    """

    # this __init__ needs to exist as otherwise passing a null resolution
    # will create an error that the object has been
    # given too many parameters at instantiation time.
    def __init__(self, *ignore: Any):
        # takes arguments and ignores them entirely
        pass

    def apply(self, FQt, t, Q):
        # pylint: disable=arguments-renamed
        # does not apply resolution
        return FQt

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them;
        NullResolution is represented as {None} to match other objects.
        """

        return "Resolution{None}"
