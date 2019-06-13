"""Module for storing properties related to atoms"""

"""
Atomic masses are taken from IUPAC 2013.  In instances where IUPAC specifies a
range of accepted values rather than a single value, the mean is taken.  All
values are in amu.
"""

MASS = {
    'H':1.00798,
    'He':4.0026,
    'Li':6.941,
    'Be':9.01218,
    'B':10.811,
    'C':12.0107,
    'N':14.0067,
    'O':15.99903,
    'F':18.998403,
    'Ne':20.1797,
    'Na':22.98977,
    'Mg':24.305,
    'Al':26.98154,
    'Si':28.0855,
    'P':30.97376,
    'S':32.06,
    'Cl':35.453,
    'K':39.0983,
    'Ar':39.948,
    'Ca':40.08,
    'Sc':44.9559,
    'Ti':47.9,
    'V':50.9415,
    'Cr':51.996,
    'Mn':54.938,
    'Fe':55.847,
    'Ni':58.7,
    'Co':58.9332,
    'Cu':63.546,
    'Zn':65.38,
    'Ga':69.72,
    'Ge':72.59,
    'As':74.9216,
    'Se':78.96,
    'Br':79.904,
    'Kr':83.8,
    'Rb':85.4678,
    'Sr':87.62,
    'Y':88.9059,
    'Zr':91.22,
    'Nb':92.9064,
    'Mo':95.94,
    'Ru':101.07,
    'Rh':102.9055,
    'Pd':106.4,
    'Ag':107.868,
    'Cd':112.41,
    'In':114.82,
    'Sn':118.69,
    'Sb':121.75,
    'I':126.9045,
    'Te':127.6,
    'Xe':131.3,
    'Cs':132.9054,
    'Ba':137.33,
    'La':138.9055,
    'Ce':140.12,
    'Pr':140.9077,
    'Nd':144.24,
    'Sm':150.4,
    'Eu':151.96,
    'Gd':157.25,
    'Tb':158.9254,
    'Dy':162.5,
    'Ho':164.9304,
    'Er':167.26,
    'Tm':168.9342,
    'Yb':173.04,
    'Lu':174.967,
    'Hf':178.49,
    'Ta':180.9479,
    'W':183.85,
    'Re':186.207,
    'Os':190.2,
    'Ir':192.22,
    'Pt':195.09,
    'Au':196.9665,
    'Hg':200.59,
    'Tl':204.37,
    'Pb':207.2
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
