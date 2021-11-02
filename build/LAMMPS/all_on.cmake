# preset that turns on all existing packages. using the combination
# this preset followed by the nolib.cmake preset should configure a
# LAMMPS binary, with as many packages included, that can be compiled
# with just a working C++ compiler and an MPI library.

set(ALL_PACKAGES ASPHERE BODY CLASS2 COLLOID COMPRESS CORESHELL DIPOLE
        GRANULAR KSPACE LATTE MANYBODY MC MISC MESSAGE MOLECULE
        OPT PERI POEMS PYTHON QEQ RIGID SHOCK SNAP SPIN
        SRD
        USER-AWPMD USER-BOCS USER-CGDNA USER-CGSDK
        USER-DIFFRACTION USER-DPD USER-DRUDE USER-EFF USER-FEP
        USER-H5MD USER-LB USER-MANIFOLD USER-MEAMC USER-MESO
        USER-MGPT USER-MISC USER-MOFFF USER-MOLFILE USER-NETCDF USER-OMP
        USER-PHONON USER-PTM USER-QTB
        USER-REAXC USER-SDPD USER-SMD USER-SMTBQ USER-SPH
        USER-TALLY USER-UEF USER-YAFF)

foreach(PKG ${ALL_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()

