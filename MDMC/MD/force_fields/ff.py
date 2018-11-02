"""A module for defining force fields that can be applied to a universe

Each force field consists of a combination of interaction functions, and also
the values of the parameters within these functions.  In this instance water
models (such as SPCE and TIP3P) are also defined as force fields, even though
the parameter sets are restricted to describing water.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-4 17:38:48"""

from abc import ABCMeta,abstractmethod


class ForceField:

    """
    Abstract class defining a force field

    For each interaction type that it uses (non-bonded, bonds, bond angles etc),
    a force field must define the interaction function (LJ, harmonic etc).  It
    must also define the parameters for each of these functions.
    """

    __metaclass__ = ABCMeta

    def __init__(self, interactions):

        """
        Arguments:
        interactions - a list of interactions
        """

        for interaction in interactions:
            self.parameterize_interaction(interaction)

    def parameterize_interaction(self, interaction):

        """
        Parameterizes the interaction with the parameters specified in the
        interaction dictionary

        Arguments:
        interaction - a subclass of MDMC.MD.structural_units.Interaction
        """

        int_type = type(interaction)
        elements = interaction.element_tuple()
        try:
            interaction.function = self.interaction_dictionary[
                (int_type, elements)]
            interaction.function_name = interaction.function.name
            interaction.function.set_params_interactions(interaction)
        except KeyError:
            raise KeyError("This force field does not have defined interactions"
                           " for these element types")
