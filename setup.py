from setuptools import find_packages, setup


setup(
    name="sparcl-st",
    version="1.1.0",
    description="SpaRCL for multi-slice spatial transcriptomics integration",
    packages=find_packages(where="SpaRCL_GitHub"),
    package_dir={"": "SpaRCL_GitHub"},
    python_requires=">=3.9",
)
