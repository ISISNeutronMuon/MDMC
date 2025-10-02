"""Unit tests for the methods of fqt.py, fqt_coh.py, fqt_incoh.py
"""

import pytest

from MDMC.trajectory_analysis.observables.fqt import FQt as FQt
from MDMC.trajectory_analysis.observables.fqt_coh import FQtCoherent
from MDMC.trajectory_analysis.observables.fqt_incoh import FQtIncoherent
from MDMC.MD.structures import Atom

@pytest.fixture
def FQt_total():

    """
    Returns
    -------
    FQt
        An initiliazed but unmodified FQt
    """

    return FQt()

@pytest.fixture
def FQt_coh():

    """
    Returns
    -------
    PairDistributionFunction
        An initiliazed but unmodified PairDistributionFunction
    """

    return FQtCoherent()

@pytest.fixture
def FQt_incoh():

    """
    Returns
    -------
    PairDistributionFunction
        An initiliazed but unmodified PairDistributionFunction
    """

    return FQtIncoherent()


class MockTrajectory():
    
    def __init__(self):
        self.element_set = []
        self.element_list = []
    
    @property
    def element_set(self):
        return  self._element_set
    
    @element_set.setter
    def element_set(self,value):
        self._element_set = value
        
    @property
    def n_atoms(self):
        return self._n_atoms
    
    @n_atoms.setter
    def n_atoms(self,value):
        self._n_atoms = value
        
    @property
    def element_list(self):
        return self._element_list
    
    @element_list.setter
    def element_list(self,value):
        self._element_list = value
    
    def exportAtom(self, atom_number):
        return Atom(self.element_list[atom_number])
    

@pytest.mark.parametrize('element_set, expected',
                         [(['Ar[36]'],{'36-Ar': { 'coh':24.9, 'incoh':0.0}}),
                           (['Ar'], {'Ar': { 'coh':1.909, 'incoh':1.3380930871145784}}),
                           (['C[13]'], {'13-C': { 'coh':6.19, 'incoh':0.5201570947860099}})])

def test_FQt_total_set_weights(FQt_total, element_set, expected):

    """
    Tests that the correct weights of isotopes are determined within FQt.
    """
    FQt_total._trajectory = MockTrajectory()
    FQt_total._trajectory.element_set = element_set
    FQt_total._set_weights()

    assert FQt_total.weights == expected
    
@pytest.mark.parametrize('element_set, expected',
                         [(['Ar[36]'],{'36-Ar': 24.9}),
                           (['Ar'], {'Ar': 1.909}),
                           (['C[13]'], {'13-C': 6.19})])
def test_FQt_coh_set_weights(FQt_coh, element_set, expected):

    """
    Tests that the correct weights of isotopes are determined within FQtCoherent.
    """
    FQt_coh._trajectory = MockTrajectory()
    FQt_coh._trajectory.element_set = element_set
    FQt_coh._set_weights()

    assert FQt_coh.weights == expected

@pytest.mark.parametrize('n_atoms, element_set, expected',
                         [(3,['Ar[36]'],[0.0, 0.0, 0.0]),
                          (3, ['Ar'], [1.7904931097838228, 1.7904931097838228, 1.7904931097838228]),
                          (2, ['C[13]'], [0.2705634032562221, 0.2705634032562221])])
def test_FQt_incoh_set_weights(FQt_incoh, n_atoms, element_set, expected):

    """
    Tests that the correct weights of isotopes are determined within FQtIncoherent.
    """
    FQt_incoh._trajectory = MockTrajectory()
    FQt_incoh._trajectory.n_atoms = n_atoms
    FQt_incoh._trajectory.element_set = element_set
    FQt_incoh._trajectory.element_list = [element_set[0] for _ in range(n_atoms)]
    FQt_incoh._set_weights()
    
    assert FQt_incoh.weights == expected
