#!/bin/bash 

# build the documentation, which includes testing that the code in the tutorials runs

docker pull mdmc/mdmc:ci-$BRANCH && docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:ci-$BRANCH /bin/bash -c  "cd $(pwd) && apt-get update && apt-get install pandoc -y && pip3 install . && make -d -C $(pwd)/doc html" && exit 0 || \
docker pull mdmc/mdmc:latest && docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update && apt-get install pandoc -y && pip3 install . && make -d -C $(pwd)/doc html" && exit 0 || exit 1
