@echo off
echo ================================================
echo   NeuralForge GUI Tester
echo ================================================
echo.
echo Starting GUI application...
echo.

python tests\gui_test.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start GUI
    echo.
    echo Installing PyQt6...
    pip install PyQt6
    echo.
    echo Retrying...
    python tests\gui_test.py
)

pause
