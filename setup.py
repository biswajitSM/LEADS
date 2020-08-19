import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "leads",
    version = "0.0.1",
    author = "Biswajit Pradhan",
    author_email = "biswajitp145@gmail.com",
    description = "Kymogram analysis for l-DNA tethered on surface.",
    long_description = long_description,
    long_description_ntent_type = "text/markdown",
    url = "https://github.com/biswajitSM/DNA-loop-Assay",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires = '>=3.8'
)