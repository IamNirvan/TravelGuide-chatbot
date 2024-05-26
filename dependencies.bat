@echo off

REM Activate virtual environment
echo activating virtual environment (env)
call env\Scripts\activate

REM Install dependencies
echo installing dependencies
pip install nltk
pip install numpy
pip install tensorflow

echo initialization completed
