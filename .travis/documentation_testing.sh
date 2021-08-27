####### This script tests whether the Docker image needs to be rebuilt
####### i.e. if there are changes to requirements.txt or the Dockerfile

# the workflow of this script is the following:
# 1a. if the job is a PR, and not a cron job, test whether changes have been made to the doc folder of the PR branch.
# 1b. if so, test the documentation builds correctly.
# 1c. if not, do not test and return a success.

# 2a. if the job is a cron job, test the documentation.
# 2b. if the documentation test is successful, deploy it as a PR to our Github Pages repository.
# 2c. if not successful, return an error.

#!/bin/bash 

# the type != cron line is to avoid giving unnecessary and contradictory output messages.
if ${TRAVIS_EVENT_TYPE}!=cron
### PR testing here
then  
  if ! git diff remotes/origin/${TRAVIS_BRANCH} remotes/origin/${TRAVIS_PULL_REQUEST_BRANCH} --name-only -- ./doc | read REPLY
  then
    echo; echo "Documentation does not require testing." 
    exit 0;
  else
    echo; echo "Documentation requires testing."
    docker pull mdmc/mdmc:latest
    docker run -t --mount type=bind,source=$(pwd),target=$(pwd) mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html"
  fi
  
  ### cron job testing & deployment here
  else 
      # testing - the `|| exit 1` at the end of the test line will exit the test and not deploy if the test build fails.
    echo; echo "Starting cron job doc testing and deployment."
    docker run -t --mount type=bind,source=$(pwd),target=$(pwd) mdmc/mdmc:latest /bin/bash -c  "cd $(pwd) && apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16 . && make -d -C $(pwd)/doc html" || exit 1
    # deployment to a 'travis' github branch
    echo; echo "Test successful. Deploying."
    cd ..
    git clone https://github.com/MDMCproject/MDMCproject.github.io
    cp ./MDMCv0.2_pilot/doc/_build/html/* ./MDMCproject.github.io/
    cd ./MDMCproject.github.io
    git checkout -B travis --track origin/travis 
    git add * && git commit -m "Travis weekly documentation update" && git push

 