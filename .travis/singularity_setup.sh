####### This script builds Singularity for Travis.
####### Originally from https://github.com/singularityhub/travis-ci

#!/bin/bash -ex

sudo sed -i -e 's/^Defaults\tsecure_path.*$//' /etc/sudoers

# Install Singularity

SINGULARITY_BASE="${GOPATH}/src/github.com/sylabs/singularity"
MAX_ENGINE_CONFIG_CHUNK=1000
ENGINE_CONFIG_CHUNK_ENV=1000
ENGINE_CONFIG_ENV=1000
MAX_CHUNK_SIZE=1000
MAX_ENGINE_CONFIG_SIZE=1000
export PATH="${GOPATH}/bin:${PATH}"

mkdir -p "${GOPATH}/src/github.com/sylabs"
cd "${GOPATH}/src/github.com/sylabs"

wget https://github.com/hpcng/singularity/releases/download/v3.8.0/singularity-3.8.0.tar.gz
tar -xzf singularity-3.8.0.tar.gz
cd singularity-3.8.0
./mconfig -v -p /usr/local
make -j `nproc 2>/dev/null || echo 1` -C ./builddir all
sudo make -C ./builddir install
