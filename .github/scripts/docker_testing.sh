#!/bin/bash 

####### This script tests whether the source code needs to be tested
####### i.e. if there are changes to requirements.txt or the MDMC folder

# the workflow of this script is the following:
# 1. check if the PR branch has a different requirements.txt or MDMC folder to the master branch
# 2. if it does, test MDMC in mdmc/mdmc:ci-[branch] if it exists or mdmc/mdmc:latest.
# 3. if it doesn't, exit.

# $TESTSET is defined in the main CI .yml script

echo "$DOCKER_PASSWORD" | docker login -u "mdmc" --password-stdin # this login circumvents the Docker IP rate limit for anonymous users
if ! git diff remotes/origin/master --name-only | grep 'MDMC/\|requirements.txt'
then
  echo "Source code does not require testing."
  exit 0 
else
  echo "Source code requires testing."
  # read this run block as 'attempt to pull the built docker container and test on it; if it fails (i.e. no container was built) then just test on latest instead'.
  docker pull mdmc/mdmc:ci-$BRANCH && docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:ci-$BRANCH python3 -m pytest "$OPTIONS"; [ $? -eq 0 ] && exit 0 || \
  docker pull mdmc/mdmc:latest &&  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest python3 -m pytest "$OPTIONS"; [ $? -eq 0 ] && exit 0 || exit 1
fi
docker logout
exit 0
