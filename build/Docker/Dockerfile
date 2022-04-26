# Use an official Python runtime as a parent image
FROM python:3.9.6-slim-buster

# Define environment variable
ENV NAME World
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH /lammps/build/
ENV PYTHONPATH ${PYTHONPATH}:/lammps/build/
ENV PYTHONUNBUFFERED 1

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
COPY .dev_requirements.txt .dev_requirements.txt 
RUN pip3 install -r .dev_requirements.txt
RUN rm .dev_requirements.txt

# Install dependencies
# [if this fails, you may be running Docker from the wrong directory. Go to
# the main directory and build the image with
# "docker build -f $(pwd)/build/Docker/Dockerfile ."]
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

# create a file in root directory called TIMESTAMP, containing
# the time and date the image was created, as well as its dependency versions.
# This is useful for troubleshooting docker image issues.
RUN echo "This docker image was created on: " > TIMESTAMP && \
echo `date` >> TIMESTAMP && \
echo "With the following packages: " >> TIMESTAMP && \
pip3 list >> TIMESTAMP

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

# get DLPOLY
RUN git clone https://gitlab.com/ccp5/dl-poly.git
RUN cmake -S dl-poly -Bbuild-dlpoly -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=off -DCMAKE_INSTALL_PREFIX=/usr/                                                                                                  
RUN cd build-dlpoly && make install
 
# Start in /home
WORKDIR /home

# Create aliases
RUN echo alias python="python3" > ~/.bash_aliases
RUN echo alias pip="pip3" >> ~/.bash_aliases
