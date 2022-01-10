####### This script builds Singularity for Travis.
####### Originally from https://github.com/singularityhub/travis-ci

#!/bin/bash -ex
# sudo resets $PATH for security reasons, so this is a workaround
# note you should never execute this line outside of a VM
sudo sed -i -e 's/^Defaults\tsecure_path.*$//' /etc/sudoers

# Install Singularity

SINGULARITY_BASE="${GOPATH}/src/github.com/sylabs/singularity"
export PATH="${GOPATH}/bin:${PATH}"
export SVER="3.8.1" #SVER is Singularity VERsion

sudo mkdir -p "${GOPATH}/src/github.com/sylabs"
cd "${GOPATH}/src/github.com/sylabs"

sudo wget https://github.com/hpcng/singularity/releases/download/v${SVER}/singularity-${SVER}.tar.gz
sudo tar -xzf singularity-${SVER}.tar.gz
cd singularity-${SVER}
./mconfig -v -p /usr/local
make -j `nproc 2>/dev/null || echo 1` -C ./builddir all
sudo make -C ./builddir install
