.. _dev_doc_container-label:

Containers
==========

What are containers and why do we use them?
-------------------------------------------

You can find a lot of information about containers on the internet, which may be overwhelming. Here containers are explained, in very simple terms, by an analogy, and in the context of how we use containers. Say our software is a hamster, and other people would like to have our hamster in their house. However, letting a hamster loose in an arbitrary house can cause problems!

We have two options. Either:
 
 * attempt to train the hamster to behave properly in any given house. Or;
 
 * build a tiny house (a hamster cage), which contains everything the hamster needs to live and function well (water, the correct food, etc.) and give anyone who wants the hamster the hamster *and* its cage, so the hamster will be in the same conditions no matter who's house it's in.
 
The container is the hamster cage; a sort of 'mini computer inside the computer'. For MDMC in particular, this container runs an appropriate Linux environment and contains all our dependency software (such as Python, various libraries, and molecular dynamics engine). An example of why we use containers is because the molecular dynamics engine (LAMMPS) can be complex to install for an arbitrary user, and with the settings and version needed by MDMC. By using a container, we can develop MDMC exclusively for one environment, while still having it function as intended on other environments/operating systems.

A container 'image' (see the next section) is a blueprint that the container software uses to build a container. When you use MDMC in a container, the docker-compose file is pulling a copy of the container image from the internet and uses it to create the container for MDMC on your machine.
 
The Docker image
----------------

The 'regular' way that users can access MDMC is via `Docker <https://www.docker.com/>`_ . The Docker container also lies at the core of our Singularity container; the Singularity container is built from the Docker image. This is mainly for ease of development, as we only need to update one image.

Our Docker image is on Docker Hub as ``mdmc/mdmc``. We use three tags for development (as of this writing):

* ``mdmc/mdmc:latest``; the version is used by e.g. docker-compose to install MDMC. This is the 'production' image.

* ``mdmc/mdmc:experimental``; if you are making manual changes to the image, it's useful to push it to this tag so you can swap it in and out vs other images, as well as let other developers and Travis use it.

* ``mdmc/mdmc:ci-[branch]``; when Github Actions automatically builds a new image (see below), it is pushed to this tag, where [branch] is the name of the branch the image was built on. 

Building the Docker image
-------------------------

As just mentioned, Github Actions automatically builds and tests new images. It does so ONLY IF files in the ``/build/Docker`` directory are changed, or if ``requirements.txt`` is changed. In this case, the pull request testing will automatically detect these changes, rebuild the image from the Dockerfile, test it, and then push it to ``mdmc/mdmc:ci-[branch]`` where [branch] is the name of the branch the image was built on. The mdmc/mdmc:travis image can then be pushed to the mdmc/mdmc:latest when required.

If you need to rebuild the Docker image manually, go to the main directory for the source code then execute the command:

.. code-block:: bash

  docker build -t mdmc/mdmc:experimental -f ./build/Docker/Dockerfile .
  
which will build the image and give it the tag mdmc/mdmc:experimental. Note that to push it to Docker hub, you need to be logged in as the mdmc user.

Please do not rebuild the Docker container using command line arguments (add them to the Dockerfile instead) or rebuild the container without updating the Dockerfile in the repository. This can cause issues and unintended behaviour, as well as making the container non-reproducible by others.

To push the Dockerhub experimental image to latest, do the following:

.. code-block:: bash

  docker pull mdmc/mdmc:experimental
  docker tag mdmc/mdmc:experimental mdmc/mdmc:latest
  docker push mdmc/mdmc:latest

and the same with mdmc/mdmc:ci-[branch] tags to push a CI image.
