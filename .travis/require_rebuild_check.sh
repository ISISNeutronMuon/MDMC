####### This script tests whether the Docker image needs to be rebuilt
####### i.e. if there are changes to requirements.txt or the Dockerfile

#!/bin/bash 

if ! git diff ${TRAVIS_BRANCH} ${TRAVIS_PULL_REQUEST_BRANCH} --name-only -- requirements.txt | read REPLY && \
  ! git diff ${TRAVIS_BRANCH} ${TRAVIS_PULL_REQUEST_BRANCH} --name-only -- ./build/Docker | read REPLY
then
  echo; echo "Docker file does not require rebuilding." 
  exit 1; # fails harmlessly
else
  echo; echo "Docker file requires rebuilding."
  REBUILD=true # sets rebuild variable to true so other docker test does not run
  exit 0; # continues docker rebuild script 
fi
