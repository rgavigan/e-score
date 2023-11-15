@echo off

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
pip install -r requirements.txt

echo Virtual environment created and activated. Dependencies installed.
