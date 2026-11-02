@echo off
rem Copyright (c) 2026 Jiayu Li, WELAB, Zhejiang University.
rem Author : Jiayu Li
rem Unit   : WELAB, Zhejiang University
rem Email  : li.jia.yu@foxmail.com
rem Date   : 2026-11-02
setlocal
cd /d "%~dp0"
if "%PYTHON%"=="" set "PYTHON=python"

"%PYTHON%" daxe.py --mode both --iters 60 --eta 3e-4 ^
    --raw raw\vessel.tif --seq seq\mask.seq --out rlt
if errorlevel 1 (
    echo [DAXE] failed with code %errorlevel%
    exit /b %errorlevel%
)
echo [DAXE] all done, results in rlt\
endlocal
