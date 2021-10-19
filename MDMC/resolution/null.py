from MDMC.resolution.resolution import Resolution


class NullResolution(Resolution):
    """
    The null object for the Resolution class.
    Used when there is no resolution to apply.
    """

    # this __init__ needs to exist as otherwise passing a null resolution will create an error that the object
    # has been given too many parameters at instantiation time.
    def __init__(self, *ignore):
        # takes arguments and ignores them entirely
        pass

    def apply(self, array, x):
        # does not apply resolution
        return array

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them;
        NullResolution is represented as {None} to match other objects.
        """

        return "Resolution{None}"
