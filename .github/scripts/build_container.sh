#!/bin/bash

# this script detects if a new docker container needs building, and if so, builds it.

echo "$DOCKER_PASSWORD" | docker login -u "mdmc" --password-stdin # this login circumvents the Docker IP rate limit for anonymous users

TOKEN=$(curl -s -H "Content-Type: application/json" -X POST -d '{"username": "'${UNAME}'", "password": "'${DOCKER_PASSWORD}'"}' https://hub.docker.com/v2/users/login/ | jq -r .token)


# full image rebuild if MD engines have changed
if git diff remotes/origin/master --name-only | grep 'build/Docker/Dockerfile.engines'
then
  echo "Rebuilding Dockerfile including base."
  docker build -t mdmc/engines:ci-$BRANCH -f "$(pwd)"/build/Docker/Dockerfile.engines . || exit 1
  docker push mdmc/engines:ci-$BRANCH
  docker build -t mdmc/mdmc:ci-$BRANCH -f "$(pwd)"/build/Docker/Dockerfile.mdmc --build-arg BASE_IMAGE=mdmc/engines:ci-$BRANCH . || exit 1
  docker push mdmc/mdmc:ci-$BRANCH
  docker logout
  exit 0
fi


# mdmc/mdmc image rebuild without base image changes
if ! git diff remotes/origin/master --name-only | grep 'build/Docker/Dockerfile.mdmc\|requirements.txt\|pyproject.toml'
then
  echo "Docker file does not require rebuilding."

  # Delete docker tags if not needed.
  if $(docker manifest inspect mdmc/mdmc:ci-$BRANCH > /dev/null)
  then
      curl -s -X DELETE -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/mdmc/engines/tags/ci-${BRANCH}/
      curl -s -X DELETE -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/mdmc/mdmc/tags/ci-${BRANCH}/
  fi
else
  echo "Docker file requires rebuilding."
  docker build -t mdmc/mdmc:ci-$BRANCH -f "$(pwd)"/build/Docker/Dockerfile.mdmc . || exit 1
  docker push mdmc/mdmc:ci-$BRANCH
fi
docker logout
exit 0
