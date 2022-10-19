#!/bin/bash
#Script will delete all images in all repositories of your docker hub account which are older than 60 days
set -e
echo

# set username and password
UNAME="mdmc"
UPASS="${DOCKER_PASSWORD}"

# get token to be able to talk to Docker Hub
TOKEN=$(curl -s -H "Content-Type: application/json" -X POST -d '{"username": "'${UNAME}'", "password": "'${UPASS}'"}' https://hub.docker.com/v2/users/login/ | jq -r .token)
echo "Token Retrieved!"

echo "List of Repositories in ${UNAME} Docker Hub account:"
REPO_LIST=$(curl -s -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/${UNAME}/?page_size=10000 | jq -r '.results|.[]|.name')
echo $REPO_LIST
echo

# build a list of all images & tags
for i in ${REPO_LIST}
do
  # get tags for repo
  IMAGE_TAGS=$(curl -s -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/${UNAME}/${i}/tags/?page_size=10000 | jq -r '.results|.[]|.name')

  # build a list of images from tags
  for j in ${IMAGE_TAGS}
  do
    # add each tag to list
    FULL_IMAGE_LIST="${FULL_IMAGE_LIST} ${UNAME}/${i}:${j}"
  done
done

echo "List of all docker images in ${UNAME} Docker Hub account:"
for i in ${FULL_IMAGE_LIST}
do
  echo ${i}
done
echo

echo "Identifying and deleting images which are older than 60 days in ${UNAME} docker hub account:"

for i in mdmc
do
  # get tags for repo
  IMAGE_TAGS=$(curl -s -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/${UNAME}/${i}/tags/?page_size=10000 | jq -r '.results|.[]|.name')

  # build a list of images from tags
  for j in ${IMAGE_TAGS}
  do
    echo "Tag Name: ${UNAME}/${i}:${j}"

    updated_time=$(curl -s -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/${UNAME}/${i}/tags/${j}/?page_size=10000 | jq -r '.last_updated')
    echo "Last Updated: $updated_time"

    datetime=$updated_time
    timeago='60 days ago'

    dtSec=$(date --date "$datetime" +'%s')
    taSec=$(date --date "$timeago" +'%s')

    echo "INFO: Last Updated Time In Seconds=$dtSec, 60 Days Ago In Seconds=$taSec"

           if [ $dtSec -lt $taSec ]
           then
              echo "This image ${UNAME}/${i}:${j} is older than 60 days, deleting this image"
              ## Please uncomment below line to delete docker hub images of docker hub repositories
              curl -s  -X DELETE  -H "Authorization: JWT ${TOKEN}" https://hub.docker.com/v2/repositories/${UNAME}/${i}/tags/${j}/
           else
              echo "This image ${UNAME}/${i}:${j} is within 60 days time range, keep this image"
           fi
           echo
  done
done

echo "Script execution ends"
