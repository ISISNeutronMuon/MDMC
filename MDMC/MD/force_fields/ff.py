"""A module for defining force fields that can be applied to a universe

Each force field consists of a combination of interaction functions, and also
the values of the parameters within these functions.  In this instance water
models (such as SPCE and TIP3P) are also defined as force fields, even though
the parameter sets are restricted to describing water.  Each force field module
is self contained, although adding a new force field may require changes to the
MD engine facades, so that a correspondence is established between the MDMC
force field and the MD engine equivalent.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-4 17:38:48"""

from abc import ABCMeta, abstractproperty


class ForceField:

    """
    Abstract class defining a force field

    For each interaction type that it uses (non-bonded, bonds, bond angles etc),
    a force field must define the interaction function (LJ, harmonic etc).  It
    must also define the parameters for each of these functions.
    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def interaction_dictionary(self):

        """
        Returns a dictionary with keys of (Interaction:Elements) where Elements
        is an ordered tuple of elemental symbols, and values of interaction
        Functions.
        """

        raise NotImplementedError

    def parameterize_interactions(self, interactions):

        """
        Parameterizes the interactions with the parameters speicifed in the
        interaction dictionary

        Arguments:
        interactions - a list of interactions
        """

        for interaction in interactions:
            self._parameterize_interaction(interaction)

    def _parameterize_interaction(self, interaction):

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
            interaction.function.set_params_interactions(interaction)
        except KeyError:
            raise KeyError("This force field does not have defined interactions"
                           " for these element types")
