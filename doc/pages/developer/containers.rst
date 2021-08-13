.. _dev_doc_container-label:

Containers
==========

What are containers and why do we use them?
-------------------------------------------

What a container is will be explained, in simplest terms, by an analogy. Say our software is a hamster, and other people would like to have our hamster in their house. However, letting a hamster loose in an arbitrary house can cause problems (e.g. the hamster might not have a food bowl, or might get stuck behind a sofa).

We have two options. Either:
 
 * we train the hamster to behave properly in any given house. For simpler hamsters this is easy, but for ones with more complex needs it can be a lot of work. Or;
 
 * we build a tiny house (a hamster cage), which contains everything the hamster needs to live and function well (water, the correct food, etc.) and give anyone who wants the hamster the hamster *and* its cage, so the hamster will be in the same conditions no matter who's house it's in.
 
The container is the hamster cage; a sort of 'mini computer inside the computer'. For MDMC in particular, this container runs Ubuntu 18.04 and contains all our dependency software (such as Python, various libraries, and the molecular dynamics engine). The reason we use containers is because our molecular dynamics engine (LAMMPS) is complex to install, particularly on Windows and Mac OS. By using a container, we can develop MDMC exclusively for Linux in a consistent environment and not worry about any quirks of other operating systems, while still having it function exactly as intended on any operating system. Furthermore, we save users the trouble of having to install LAMMPS themselves.

A container 'image' (which will be discussed in the next section) is a blueprint that the container software uses to build a container. When you use MDMC in a container, the docker-compose file is pulling a copy of the container image from the internet and using it to create the container for MDMC on your machine.
 
The Docker image
----------------

The 'regular' way that users can access MDMC is via our `Docker <https://www.docker.com/>`_ container. The Docker container also lies at the core of our Singularity container; the Singularity container is built from the Docker image. This is mainly for ease of development, as we only need to update one image.

Our Docker image is on Docker Hub as ``mdmc/mdmc``. We use three tags for development:

* ``mdmc/mdmc:latest``; the version used by docker-compose to install MDMC. This is the 'production' image.

* ``mdmc/mdmc:experimental``; if you are making manual changes to the image, it's useful to push it to this tag so you can swap it in and out vs other images, as well as let other developers and Travis use it.

* ``mdmc/mdmc:travis``; when Travis CI automatically builds a new image (see below), it is pushed to this tag. 

Building the Docker image
-------------------------

As just mentioned, Travis CI, our testing and continuous integration service, automatically builds and tests new images. It does so ONLY IF files in the ``/build/Docker`` directory are changed, or if ``requirements.txt`` is changed. In this case, the pull request testing will automatically detect these changes, rebuild the image from the Dockerfile, test it, and then push it to ``mdmc/mdmc:travis``. The mdmc/mdmc:travis image should regularly be pushed to the mdmc/mdmc:latest when required.

If you need to rebuild the Docker image manually, go to the main directory for the source code then execute the command:

.. code-block:: bash

  docker build -t mdmc/mdmc:experimental -f ./build/Docker/Dockerfile .
  
which will build the image and give it the tag mdmc/mdmc:experimental. Note that to push it to Docker hub, you need to be logged in as the mdmc user.

Please do not rebuild the Docker container using command line arguments (add them to the Dockerfile instead) or rebuild the container without updating the Dockerfile in the repository. This can cause issues and unintended behaviour, as well as making the container non-reproducible by others.
