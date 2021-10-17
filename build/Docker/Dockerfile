# Use an official Python runtime as a parent image
FROM python:3.9.6-slim-buster

# Define environment variable
ENV NAME World
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH /lammps/build/
ENV PYTHONPATH ${PYTHONPATH}:/lammps/build/

# Install pip, hdf5, netcdf, bash, git, fortran and blas (for numpy and
# scipy), and cmake for LAMMPS
RUN apt-get update && apt-get install --upgrade -y \
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

# Install dependencies
# [if this fails, you may be running Docker from the wrong directory. Go to
# the main directory and build the image with
# "docker build -f $(pwd)/build/Docker/Dockerfile ."]
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

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
          -D LAMMPS_EXCEPTIONS=yes \
          -D DOWNLOAD_VORO=on \
          -D PKG_NETCDF=yes \
          -D PKG_VORONOI=on \
          -D PKG_NETCDF=yes \
          -D BUILD_LAMMPS_TOOLS=yes \
          -D BUILD_OMP=yes \
          -D PKG_MOLECULE=yes \
          -D PKG_PYTHON=yes \ 
          -D PKG_MOLFILE=yes \
          -D PKG_VTK=yes \
          ../cmake
RUN make
RUN make install-python

# Start in /home
WORKDIR /home

# Create aliases
RUN echo alias python="python3" > ~/.bash_aliases
RUN echo alias pip="pip3" >> ~/.bash_aliases
