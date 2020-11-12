@echo off

call conda activate leads-env
call python -m leads.crop_images_batch
PAUSE
