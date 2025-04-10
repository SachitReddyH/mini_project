@echo off
echo Starting Indian Sign Language Translator...
echo The application will run in the background.
echo To stop the application, close this window or press Ctrl+C.

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

:: Start the application in the background
start /B python run_app.py --host 127.0.0.1 --port 5000 > logs\console.log 2>&1

echo Application started. Check logs\console.log for output.
echo You can access the application at http://127.0.0.1:5000
echo.
echo Press any key to stop the application...
pause > nul

:: Find and terminate the Python process running our app
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /v ^| find "run_app.py"') do taskkill /F /PID %%a
echo Application stopped. 