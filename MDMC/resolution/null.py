from MDMC.resolution.resolution import Resolution


class NullResolution(Resolution):
    """
    The null object for the Resolution class.
    Used when there is no resolution to apply.
    """

    def apply(self, x, array):
        return array

