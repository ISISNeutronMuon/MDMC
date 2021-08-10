####### This script tests whether the Docker image needs to be rebuilt
####### i.e. if there are changes to requirements.txt or the Dockerfile

#!/bin/bash -ex

if ! git diff remotes/origin/master master --name-only -- requirements.txt | read REPLY && \
  ! git diff remotes/origin/master master --name-only -- ./build/Docker | read REPLY
then
  echo; echo "Docker file does not require rebuilding." 
  exit 0;
fi