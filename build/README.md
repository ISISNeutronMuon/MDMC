# Docker containers

Summary and rationale
---------------------

MDMC makes heavy use of Docker containers in order to run quickly and
easily (i.e. without making users install MD engines, which takes a lot
of time and effort). The container consists of two layers:

-   `mdmc/engines`, the backend, containing MD engines and base apt-get
    dependencies

-   `mdmc/mdmc`, the frontend, containing the Python packages used by
    MDMC.

The reason we split it into two is because we cannot cache the MD
engines - every time the image is built, LAMMPS is re-compiled from the
web source. Thus by using two images, we rebuild MD engines only when
needed, and otherwise the latest-built version of `mdmc/engines` is
simply pulled into `mdmc/mdmc` as a base image. Without this two-layer
approach, we would have to spend 30 minutes recompiling LAMMPS every
time we wanted to update a Python package.
MDMC can also be accessed via Singularity - this is done via
[Singularity's Docker
interface](https://sylabs.io/guides/latest/user-guide/singularity_and_docker.html).
This interface is tested as part of MDMC's weekly tests.

`mdmc/engines`
--------------

`mdmc/engines` is built from the file `Dockerfile.engines`. Its sections
are as follows.

1.  The base image; this is the image Python 3.11.2-slim-buster, a Debian
    Buster system with Python preinstalled. This allows us fine version
    control over which Python version we are using for MDMC.

2.  `ENV` variables - these contain a few variables that MDMC needs to
    run. Most of them are OS housekeeping, or path variables used so
    that our dependencies run smoothly. Of note is the variable
    `PYTHONUNBUFFERED 1`, which runs Python in ['unbuffered'
    mode](https://docs.python.org/3.11/using/cmdline.html#envvar-PYTHONUNBUFFERED).
    This is here because when running LAMMPS in a Jupyter notebook, it
    needs to be able to capture unbuffered stdout in order to work.

3.  `apt-get` dependencies. These are pretty standard; just prerequisite
    libraries needed for our Python requirements to function.

4.  MD engines. The rest of the file (except some miscellaneous
    convenience functions at the end) gets our MD engines from the
    Internet and compiles them.

`mdmc/mdmc`
-----------

`mdmc/mdmc` is built from the file `Dockerfile.mdmc`. Its sections are
as follows.

1.  The base image; we use an `ARG` to get this. It defaults to
    `mdmc/engines:latest`, but if you want to use a different base image
    (for example, if your development branch's `mdmc/engines` changes
    and you want to use the dev branch's version), adding the option:

    `--build-arg BASE_IMAGE=mdmc/engines:TAG`

    to `docker build` will use the image `mdmc/engines:TAG` as a base
    instead.

2.  Python requirements. These are installed from a requirements
    files; the requirements file `dependencies.txt` is generated from
    the `pyproject.toml` and `requirements.txt` by `piptools`. Using
    files like this allows us to version-control both regular and
    development packages within our Docker image via Dependabot.

3.  `TIMESTAMP`; we create a file in the base directory of the image
    called TIMESTAMP. This contains the time and date that the image was
    created, and all Python packages installed within it. If you're
    having issues with the image, this is the first port of call for
    debugging; it contains all the information about how old the image
    you're using is, and what package versions are in it. If you want to
    debug the image mdmc/mdmc:TAG (where TAG is the image tag), run

    `docker run -t mdmc/mdmc:TAG cat TIMESTAMP`

    in a terminal.
