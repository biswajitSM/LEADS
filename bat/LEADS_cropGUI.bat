@echo off

call conda activate leads-env
Echo Launch dir: "%~dp0"
cd ..
Echo Current dir: "%CD%"
python -m leads.gui.crop_images_gui