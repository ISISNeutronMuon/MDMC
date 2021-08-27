####### This script tests whether the Docker image needs to be rebuilt
####### i.e. if there are changes to requirements.txt or the Dockerfile

# the workflow of this script is the following:
# 1. check if the PR branch has a different requirements.txt or Dockerfile to the master branch
# 2. if it does, rebuild the Docker container under the tag mdmc/mdmc:travis. Test the software inside this new container, then push the new container.
# 3. if it doesn't, test with the original container (mdmc/mdmc:latest)

#!/bin/bash 

if ! git diff remotes/origin/${TRAVIS_BRANCH} remotes/origin/${TRAVIS_PULL_REQUEST_BRANCH} --name-only -- requirements.txt | read REPLY && \
  ! git diff remotes/origin/${TRAVIS_BRANCH} remotes/origin/${TRAVIS_PULL_REQUEST_BRANCH} --name-only -- ./build/Docker | read REPLY
then
  echo; echo "Docker file does not require rebuilding." 
  docker pull mdmc/mdmc:latest
  docker run -t --mount type=bind,source=$(pwd),target=$(pwd) mdmc/mdmc:latest python3 -m pytest -s $(pwd)/tests/ --cov=$(pwd)/MDMC --cov-report xml
else
  echo; echo "Docker file requires rebuilding."
  docker build -t mdmc/mdmc:travis -f $(pwd)/build/Docker/Dockerfile . || exit 1 
  #docker run -t --mount type=bind,source=$(pwd),target=$(pwd) mdmc/mdmc:travis python3 -m pytest -s $(pwd)/tests/ --cov=$(pwd)/MDMC --cov-report xml
  docker run -t --mount type=bind,source=$(pwd),target=$(pwd) mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html"
fi
