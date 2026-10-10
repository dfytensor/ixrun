@echo off
set PYTHONUTF8=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
set INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include;%INCLUDE%
set LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64;%LIB%
echo === stage 1: regenerate cuda.cu (nvcc may AV from python; expected) ===
"F:\rwkv\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'E:\IXRUN'); from ixrun.cpp_engine_27b import _build_ext; _build_ext(); print('STAGE1_PY_BUILD_OK')" 2>&1 | findstr /C:"STAGE1_PY_BUILD_OK" /C:"SYNTAX" /C:"syntax" /C:"error:" /C:"error :" /C:"C2784" /C:"C2065" /C:"C2143" /C:"identifier" 
echo === stage 2: ninja from cmd ===
cd /d C:\Users\Administrator\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cu126\ixrun_cpp_q27m
ninja -v
echo BUILD_EXIT=%ERRORLEVEL%
