@echo off
echo ============================================
echo  AI Financial Analytics - Starting App
echo ============================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call ..\..\.venv\Scripts\activate.bat

echo.
echo Starting Streamlit app...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

streamlit run app_clean.py

pause
