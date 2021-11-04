build dependencies:
libeigen3-dev
libpthread-stubs0-dev

location of netcdf header:
/usr/lib/x86_64-linux-gnu/libnetcdf.so

cp /workspaces/MDMCv0.2_pilot/build/LAMMPS/all_on.cmake /lammps/cmake/presets/
cp /workspaces/MDMCv0.2_pilot/build/LAMMPS/Makefile.lammps_NETCDF /lammps/lib/netcdf/Makefile.lammps
cp /workspaces/MDMCv0.2_pilot/build/LAMMPS/Makefile.lammps_MOLFILE /lammps/lib/molfile/Makefile.lammps

export PATH=${PATH}:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu
export PYTHONPATH=${PYTHONPATH}:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu


full paths:
export PATH=/lib:/bin:/include:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu:/usr/include:/usr/bin:/usr/local/bin:/usr/include:/usr/local/include
export PYTHONPATH=/lib:/bin:/include:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/include/x86_64-linux-gnu:/usr/include:/usr/bin:/usr/local/bin:/usr/include:/usr/local/include

to compile:
cmake -C ../cmake/presets/all_on.cmake ../cmake
make -j4
make install
make install-python
