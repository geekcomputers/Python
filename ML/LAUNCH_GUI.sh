#!/bin/bash

echo "================================================"
echo "  NeuralForge GUI Tester"
echo "================================================"
echo ""
echo "Starting GUI application..."
echo ""

python3 tests/gui_test.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to start GUI"
    echo ""
    echo "Installing PyQt6..."
    pip3 install PyQt6
    echo ""
    echo "Retrying..."
    python3 tests/gui_test.py
fi
