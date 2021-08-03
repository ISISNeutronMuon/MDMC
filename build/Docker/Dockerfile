# Use an official Python runtime as a parent image
FROM ubuntu:18.04

# Define environment variable
ENV NAME World
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH /lammps/build/
ENV PYTHONPATH ${PYTHONPATH}:/lammps/build/

# Install python, pip, hdf5, netcdf, bash, git, fortran and blas (for numpy and
# scipy), and cmake for LAMMPS
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-dev \
  libhdf5-dev \
  libnetcdf-dev \
  bash \
  git \
  gfortran \
  libopenmpi-dev \
  libblas-dev \
  liblapack-dev \
  libfreetype6-dev \
  cmake \
  wget \
  python3-tk

# Install any packages required for container which are not specifically for
# running MDMC (i.e. not in requirements.txt)
RUN pip3 install --upgrade pip
RUN pip3 install setuptools --upgrade
RUN pip3 install pytest \
  pytest-cov \
  pytest-cases \
  ipython>=5.8.0 \
  jupyter

# Install dependencies (like this for testing, when it works I will link it to the .txt)
RUN pip3 install netCDF4 \
numpy \
pandas \
scipy \
numba \
ase>=3.19 \
mpi4py \
ipython \
ipykernel


# Get LAMMPS
RUN wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz
RUN mkdir /lammps \
    && tar xvzf lammps-stable.tar.gz -C /lammps --strip-components=1
RUN rm /lammps-stable.tar.gz

# Build LAMMPS from source
WORKDIR /lammps/build
RUN cmake -C ../cmake/presets/all_on.cmake \
          -C ../cmake/presets/nolib.cmake \
          -D BUILD_LIB=on \
          -D BUILD_SHARED_LIBS=on \
          -D BUILD_MPI=on \
          -D LAMMPS_EXCEPTIONS=on \
          -D DOWNLOAD_VORO=on \
          -D PKG_PYTHON=on \
          -D PKG_VORONOI=on \
          ../cmake
RUN make
RUN make install-python

# Start in /home
WORKDIR /home

# Create aliases
RUN echo alias python="python3" > ~/.bash_aliases
RUN echo alias pip="pip3" >> ~/.bash_aliases
