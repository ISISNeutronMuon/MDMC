build dependencies:
libeigen3-dev
libpthread-stubs0-dev

location of netcdf header:
/usr/lib/x86_64-linux-gnu/libnetcdf.so

all_on.cmake copy to /lammps/cmake/presets
Makefile.lammps_NETCDF copy to /lammps/lib/netcdf/Makefile.lammps
Makefile.lammps_MOLFILE copy to /lammps/lib/molfile/Makefile.lammps

export PATH=${PATH}:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu
export PYTHONPATH=${PYTHONPATH}:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu


full paths:
export PATH=/lib:/bin:/include:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu:/usr/include:/usr/bin:/usr/local/bin:/usr/include:/usr/local/include
export PYTHONPATH=/lib:/bin:/include:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu:/usr/include:/usr/bin:/usr/local/bin:/usr/include:/usr/local/include
