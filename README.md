# Loop extrusion assay by DNA and SMC (LEADS)

Single-molecule assay for DNA loop extrusion by SMC. Kymograph and kinetics from the loop assays.

## Project Details

![image](resources/kymograph_gui.png)


## Installation

It's best to use an isolated environment for this program. But it might just work with your standard/base environment if it has python>=3.7.

You can create an environment in conda by running the following line in a terminal

```sh
conda create -n leads-env python=3.8
```

Then activate the environment by

```sh
conda activate leads-env
```

You can install the LEADS module by dowloading this [repo](https://github.com/biswajitSM/LEADS/archive/master.zip). Go to the folder containing setup file in terminal and install by the following commands.

Install all the dpendencies
```sh
pip install -r requirements.txt
```
And then install the leads program
```sh
python setup.py install
```

## Usage

### From a terminal or command prompt

Activate environment in a terminal
```sh
conda activate leads-env
```

To open the gui of kymograph analysis
```sh
python -m leads.gui.kymograph_gui
```

To open the gui for cropping large number of image files in a folder
```sh
python -m leads.gui.crop_images_gui
```

### Alternative

If you are in windows os, you can double click the bat script files [LEADS_kymoGUI.bat](./bat/LEADS_kymoGUI.bat) for the kymo program and [LEADS_cropGUI.bat](./bat/LEADS_cropGUI.bat) for the cropping program.

## Contributing

Please use issues to post your bugs and send pull requests to merge your modifications/improvemnts.

All the contributors will be acknowledged here and will be included if this ends up in a publication.

## Authors & Contributors

List of contributors:

- Biswajit Pradhan (biswajitp145_at_gmail.com)
- Roman Barth

## Pubslihed articles based on this module

[SMC complexes can traverse physical roadblocks bigger than their ring size, *Cell Reports*, 2022](https://doi.org/10.1016/j.celrep.2022.111491)

[Condensin-driven loop extrusion on supercoiled DNA. *nature structural molecular biology*, 2022](https://doi.org/10.1038/s41594-022-00802-x)

[ParS-independent recruitment of the bacterial chromosome-partitioning protein ParB. *Science Advances*, 2022](https://doi.org/10.1126/sciadv.abn3299)

[The Smc5/6 complex is a DNA loop extruding motor. *Nature*(under review), 2022](https://doi.org/10.1101/2022.05.13.491800)

[Can pseudotopological models for SMC-driven DNA loop extrusion explain the traversal of physical roadblocks bigger than the SMC ring size? *bioRxiv*, 2023](https://doi.org/10.1101/2022.08.02.502451)
