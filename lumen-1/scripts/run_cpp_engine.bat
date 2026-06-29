@echo off
echo ===================================================
echo   🚀 Lumen-1 Ultra-Fast C++ Inference (llama.cpp)
echo ===================================================

REM Check if GGUF model exists
IF NOT EXIST "..\checkpoints\lumen_mistral_merged.gguf" (
    echo.
    echo [ERROR] GGUF model not found! 
    echo Please run export_model.py and quantize.py first to convert your model to GGUF format.
    echo Expected path: ..\checkpoints\lumen_mistral_merged.gguf
    echo.
    pause
    exit /b
)

REM Run the C++ engine
echo.
echo ✅ Starting C++ Engine...
echo Type your prompt below (Press Ctrl+C to exit)
echo.

REM Assuming the user has downloaded llama.cpp windows release
server.exe -m "..\checkpoints\lumen_mistral_merged.gguf" -c 2048 --port 8080

echo.
echo [INFO] C++ Server is running on http://127.0.0.1:8080
echo You can now connect to this server using your App!
pause
