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
    python_requires = '>=3.7',
    install_requires = [
        # "numpy",
        # "numba",
        # "pandas",
        # "scipy",
        # "scikit-image",
        # "matplotlib==3.4.1",
        # "PyQt5",
        # "napari==0.4.8",
        # "pyqtgraph",
        # "opencv-python",
        # "dask",
        # "pims",
        # "trackpy",
        # "tqdm",
        # "tifffile==2020.6.3",
        # "roifile==2020.5.28",
        # "PyYAML",
        # "h5py",
        # "tables",
        # "qdarkstyle",
        # "colorharmonies",
        # "XlsxWriter"
    ],
    dependency_links = ["git+https://github.com/pyqtgraph/pyqtgraph@master"]
)