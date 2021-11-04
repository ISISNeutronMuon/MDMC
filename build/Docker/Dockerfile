# Use an official Python runtime as a parent image
FROM python:3.9.6-slim-buster

# Define environment variable
ENV NAME World
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH /lammps/build/
ENV PYTHONPATH ${PYTHONPATH}:/lammps/build/
ENV PATH ${PATH}:/lammps/build/

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
  python3-tk \
  libeigen3-dev \
  libpthread-stubs0-dev

# Install any packages required for container which are not specifically for
# running MDMC (i.e. not in requirements.txt)
RUN pip3 install --upgrade pip
RUN pip3 install setuptools --upgrade
COPY .dev_requirements.txt .dev_requirements.txt 
RUN pip3 install -r .dev_requirements.txt

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

# copy over makefiles
COPY ./build/LAMMPS/all_on.cmake /lammps/cmake/presets/all_on.cmake
COPY ./build/LAMMPS/Makefile.lammps_NETCDF /lammps/lib/netcdf/Makefile.lammps
COPY ./build/LAMMPS/Makefile.lammps_MOLFILE /lammps/lib/molfile/Makefile.lammps

# Build LAMMPS from source
WORKDIR /lammps/build
RUN cmake -C ../cmake/presets/all_on.cmake \
          ../cmake
RUN make
RUN make install
RUN make install-python

# Start in /home
WORKDIR /home

# Create aliases
RUN echo alias python="python3" > ~/.bash_aliases
RUN echo alias pip="pip3" >> ~/.bash_aliases
