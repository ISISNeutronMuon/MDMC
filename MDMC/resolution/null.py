from MDMC.resolution.resolution import Resolution


class NullResolution(Resolution):
    """
    The null object for the Resolution class.
    Used when there is no resolution to apply.
    """

    def __init__(self, *ignore):
        # takes arguments and ignores them entirely
        pass

    def apply(self, array, x):
        # does not apply resolution
        return array

