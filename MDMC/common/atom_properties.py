"""Module for storing properties related to atoms

AUTHOR :    Thomas Farmer        START DATE :    04/07/2018, 11:04:45"""

"""
Atomic masses are taken from IUPAC 2013.  In instances where IUPAC specifies a
range of accepted values rather than a single value, the mean is taken.  All
values are in amu.
"""

MASS = {
    'H':1.00798,






    'O':15.99903

}


"""
Neutron scattering lengths are taken from Neutron News, Vol. 3, No. 3, 1992,
pp. 29-37.  All values are in fm.
"""

B_COH = {
    'H':-3.7390,




    'O':5.803
}

B_INCOH = {
    'H':25.27229286,

    'O':0.
}
