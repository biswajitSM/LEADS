@echo off

call cd ..
REM Remove acq environment
call conda deactivate
echo y | call conda remove -n leads-env --all
REM Install acq environment
echo y | call conda create -n leads-env python=3.8
call conda activate leads-env
echo y | call pip install -r requirements.txt
call pip install -e .
cmd /k
