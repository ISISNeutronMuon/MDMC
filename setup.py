from distutils.core import setup

setup(
    name="MDMC",
    version="0.2pilot",
    package_dir={"":"MDMC"},
    packages=[
        "src",
        "src.common",
        "src.MD",
        "src.readers",
        "src.refinement",
        "src.trajectory_analysis",
        "src.utilities"
    ],
    long_description=open("README.md").read(),
    author="Thomas Farmer",
    url="https://github.com/MDMCproject",
    install_requires=["numpy==1.8.0"]
)
