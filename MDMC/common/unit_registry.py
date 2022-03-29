"""Contains the Pint unit registry for MDMC.
Custom units etc. can be defined here."""

import pint

UREG = pint.UnitRegistry()
UREG.setup_matplotlib()
