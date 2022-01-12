#!/bin/bash 

# build the documentation, which includes testing that the code in the tutorials runs

# tests if there is a change to doc but not to ipython
if git diff master --name-only -- ./doc | read REPLY && \
! git diff master -- requirements.txt | grep  '+ipython\|+ipykernel' 
then
  echo; echo "Documentation requires testing."
  docker pull mdmc/mdmc:latest
  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html"
# tests if there is a change to ipython which affects jupyter
elif git diff remotes/origin/master remotes/origin/"$BRANCH" -- requirements.txt | grep  '+ipython\|+ipykernel'
then
  echo; echo "Documentation requires testing on new image"
  docker pull mdmc/mdmc:ci-$BRANCH
  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:ci-$BRANCH /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html"
else
  echo; echo "Documentation does not require testing."
  exit 0;
fi
