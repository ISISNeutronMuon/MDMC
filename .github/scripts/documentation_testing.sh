#!/bin/bash 

# build the documentation, which includes testing that the code in the tutorials runs

if git diff remotes/origin/master remotes/origin/"$BRANCH" --name-only -- ./doc | read REPLY && \
! git diff remotes/origin/master remotes/origin/"$BRANCH" -- requirements.txt | grep  '+ipython\|+ipykernel' # if there is a change to doc but not to ipython
then
  echo; echo "Documentation requires testing."
  docker pull mdmc/mdmc:latest
  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html"
elif git diff remotes/origin/master remotes/origin/"$BRANCH" -- requirements.txt | grep  '+ipython\|+ipykernel' # if there is a change to ipython which affects jupyter
then
  echo; echo "Documentation requires testing on new image"
  docker pull mdmc/mdmc:travis
  docker run -t --mount type=bind,source="$(pwd)",target="$(pwd)" mdmc/mdmc:travis /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html"
else
  echo; echo "Documentation does not require testing."
  exit 0;
fi
