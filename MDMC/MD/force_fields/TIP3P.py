from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import NonBonded
from MDMC.MD.interactions import NonBondedInteraction


class TIP3P(WaterModel):

    """
    TIP3P force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Charge Parameters
        q_O = -0.834        # e
        q_H = abs(q_O/2)    # e

        # LJ Parameters
        sigma = 3.151       # Ang
        epsilon = 0.6363    # kJ mol^-1

        return {
            (NonBondedInteraction, ('O',)): NonBonded(q_O, sigma, epsilon),
            (NonBondedInteraction, ('H',)): NonBonded(q_H, 0.0, 0.0),
        }
