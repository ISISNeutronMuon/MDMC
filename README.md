# MDMCv0.2_pilot
A pilot version of MDMCv0.2 to refine experimental data for simple MD models (e.g. water).

### To test

- Download and install Docker (you may need to create an account)
- Start Docker

In terminal:
- start bash in docker container of mmtk_dependencies:

`docker run -t -i tfarmer/mmtk_dependencies:latest bash`

- Use pip to install MDMC. Specify the branch after @, e.g. for refinement branch:

`pip install -e git+https://github.com/MDMCproject/MDMCv0.2_pilot.git@refinement#egg=MDMC`

- Enter GitHub username
- Enter GitHub password
