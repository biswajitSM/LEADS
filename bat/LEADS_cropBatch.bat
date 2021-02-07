@echo off

call conda activate leads-env
Echo Launch dir: "%~dp0"
cd ..
Echo Current dir: "%CD%"
call python -m leads.crop_images_batch
PAUSE
