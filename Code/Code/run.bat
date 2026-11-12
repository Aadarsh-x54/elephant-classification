@echo off
REM Quickstart script for Windows
echo === Elephant Classifier Setup ===
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Running tests...
pytest tests/test_project.py -v

echo Starting Streamlit web app...
python -m streamlit run app.py
pause

