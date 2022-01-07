#!/bin/bash 

####### This script tests whether the Docker image needs to be rebuilt
####### i.e. if there are changes to requirements.txt or the Dockerfile

# the workflow of this script is the following:
# 1. check if the PR branch has a different requirements.txt or Dockerfile to the master branch
# 2. if it does, rebuild the Docker container under the tag mdmc/mdmc:travis. Test the software inside this new container, then push the new container.
# 3. if it doesn't, test with the original container (mdmc/mdmc:latest)

# $TESTSET is defined in CI so we only need one script for both lammps and non-lammps tests in parallel

echo "$DOCKER_PASSWORD" | docker login -u "mdmc" --password-stdin # this login circumvents the Docker IP rate limit for anonymous users
if ! git diff remotes/origin/master remotes/origin/"$BRANCH" --name-only | grep 'build/\|requirements.txt'
then
  echo "Docker file does not require rebuilding." 
  docker pull mdmc/mdmc:latest
  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest python3 -m pytest -m "$TESTSET" || exit 1
else
  echo "Docker file requires rebuilding."
  # if this is lammps tests (as they run first) then build the new image
  if [ "$TESTSET" == lammps]; then
    docker build -t mdmc/mdmc:ci -f "$(pwd)"/build/Docker/Dockerfile . || exit 1 
  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:ci python3 -m pytest -m "$TESTSET" || exit 1
  docker push mdmc/mdmc:ci # pushes docker image if test was successful
fi
docker logout
