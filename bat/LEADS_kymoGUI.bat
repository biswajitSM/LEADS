@echo off

call conda activate leads-env
Echo Launch dir: "%~dp0"
cd ..
Echo Current dir: "%CD%"
call python -m leads.gui.kymograph_gui
cmd \k