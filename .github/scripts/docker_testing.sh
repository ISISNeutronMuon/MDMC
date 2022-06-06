#!/bin/bash 

# $TESTSET is defined in the main CI .yml script


# read this run block as 'attempt to pull the built docker container and test on it; if it fails (i.e. no container was built) then just test on latest instead'.
docker pull mdmc/mdmc:ci-$BRANCH && docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:ci-$BRANCH /bin/bash -c "$1";[ $? -eq 0 ] && exit 0 || \
docker pull mdmc/mdmc:latest &&  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest /bin/bash -c "$1";[ $? -eq 0 ] && exit 0 || exit 1
docker logout
exit 0
