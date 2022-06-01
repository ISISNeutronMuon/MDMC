#!/bin/bash 

# build the documentation, which includes testing that the code in the tutorials runs

# tests if there is a change to doc or packages used to build doc
if ! git diff remotes/origin/master --name-only -- ./doc | read REPLY && \
! git diff remotes/origin/master -- requirements.txt | grep  '+ipython\|+ipykernel' && \
! git diff remotes/origin/master -- .dev_requirements.txt | grep '+sphinx\|+nbsphinx\|+sphinx-rtd-theme\|+docutils'
then
  echo; echo "Documentation does not require testing."
  exit 0
else
  echo; echo "Documentation requires testing."
  docker pull mdmc/mdmc:ci-$BRANCH && docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:ci-$BRANCH /bin/bash -c  "cd $(pwd) && apt-get update && apt-get install pandoc -y && pip3 install . && make -d -C $(pwd)/doc html" && exit 0 || \
  docker pull mdmc/mdmc:latest && docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update && apt-get install pandoc -y && pip3 install . && make -d -C $(pwd)/doc html" && exit 0 || exit 1
fi
