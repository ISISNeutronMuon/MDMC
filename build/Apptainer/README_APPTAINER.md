# Here you find the description on how to create Apptainer images from def-files.

Firstly, create the image for engines by running a command:

```
sudo apptainer build apptainer_mdmc_engines.sif apptainer_mdmc_engines.def
```

Note, the right order: def-file goes before sif-file otherwise, it will overwrite the def-file with nothing.

When you created the image for your engines, you can create the image for MDMC by running the following command:

```
sudo apptainer build apptainer_mdmc.sif apptainer_mdmc.def
```

If you want to run images on HPC or just with Apptainer on any Linux system you'll need to run it like this:

```
sudo apptainer exec apptainer_mdmc.sif python3 argon.py > output.log 
```

Here we run only the last image.

If you want to enter the shell and execute commands inside it you can run the following command to enter the shell:

```
sudo apptainer run apptainer_mdmc.sif 
```

Then you execute scripts inside the shell. This container doesn't create any notebook and can be run in WSL's terminal.
Due to it's small size it can be easily executed in HPC environments.


