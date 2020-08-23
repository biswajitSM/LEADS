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
    python_requires = '>=3.8',
    install_requires = [
        "numpy>=1.18",
        "pandas==1.0.3",
        "scipy>=1.4",
        "scikit-image>=0.17",
        "PyQt5==5.15.0",
        "napari==0.3.4",
        "pyqtgraph>=0.11",
        "dask>=2.16.0",
        "PySimpleGUI",
        "pims==0.4.1",
        "tqdm==4.46.1",
        "tifffile==2020.6.3",
        "roifile==2020.5.28",
        "PyYAML"
    ]
)