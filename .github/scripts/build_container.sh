#!/bin/bash

# this script detects if a new docker container needs building, and if so, builds it.

echo "$DOCKER_PASSWORD" | docker login -u "mdmc" --password-stdin # this login circumvents the Docker IP rate limit for anonymous users
if ! git diff remotes/origin/master --name-only | grep 'build/\|requirements.txt'
then
  echo "Docker file does not require rebuilding." 
else
  echo "Docker file requires rebuilding."
  docker pull mdmc/mdmc:ci-$BRANCH && docker build --cache-from mdmc/mdmc:ci-$BRANCH -t mdmc/mdmc:ci-$BRANCH -f "$(pwd)"/build/Docker/Dockerfile . || docker pull mdmc/mdmc:latest && docker build --cache-from mdmc/mdmc:latest -t mdmc/mdmc:ci-$BRANCH -f "$(pwd)"/build/Docker/Dockerfile . || exit 1 
  docker push mdmc/mdmc:ci-$BRANCH
fi
docker logout
exit 0

