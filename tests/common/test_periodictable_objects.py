"""Tests methods dealing with periodictable objects"""

from MDMC.trajectory_analysis.observables.fqt import create_list_of_element_objects
import pytest


@pytest.mark.parametrize('elements_list, expected', [('36-Ar', '36-Ar'), ('Ar','Ar'),('Ar[36]','36-Ar')])
def test_create_list_of_element_objects(elements_list, expected):
    """
    Tests that when a list is passed to the create_list_of_element_objects method, the objects are
    created successfully with the correct signifiying strings.
    """
    
    element_object = create_list_of_element_objects([elements_list])
    
    assert str(element_object[0]) == expected
