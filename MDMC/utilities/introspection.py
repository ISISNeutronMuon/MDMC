"""A module for all utility classes and functions that use introspection

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 11:03:30"""

from inspect import stack


# TODO: Add class for generating factories, to remove repetitions

def get_calling_class(levels_up=1):

    """
    Returns the calling class of whichever object calls this function

    By default this inspects the calling class of the frame directly above on
    the stack, however the number of levels up on the stack can be specified.
    """

    try:
        levels_up += 1          # Account for this function at top of stack
        frame = stack()[levels_up][0]
        cls = type(frame.f_locals.get('self'))
        return cls
    # TODO: Test for other exceptions
    except KeyError:
        # TODO: Determine if this is the most appropriate error to raise
        raise AttributeError("Object has no calling class")
    finally:
        del frame
