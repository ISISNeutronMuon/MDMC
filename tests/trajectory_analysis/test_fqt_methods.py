from MDMC.trajectory_analysis.observables.fqt import create_list_of_element_objects

def test_create_list_of_element_objects():
    """
    Tests that when a list is passed to the create_list_of_element_objects method, the objects are
    created successfully with the correct signifiying strings.
    """
    
    elements_list = ['36-Ar', 'Ar', 'Ar[36]']
    
    atoms = create_list_of_element_objects(elements_list)
    
    assert str(atoms[0]) == '36-Ar'
    assert str(atoms[1]) == 'Ar'
    assert str(atoms[2]) == '36-Ar'
    