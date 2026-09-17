@echo off
REM ============================================================
REM Wrapper para ejecutar comandos con cachés en D:\
REM Uso:
REM   run kedro run --pipeline fetch_stock_prices
REM   run kedro viz
REM   run pytest
REM   run ruff check .
REM ============================================================

set UV_CACHE_DIR=D:\cache\uv
set PIP_CACHE_DIR=D:\cache\pip
set TORCH_HOME=D:\cache\torch
set RUFF_CACHE_DIR=D:\cache\ruff

REM Activar el entorno virtual del proyecto
call "%~dp0.venv\Scripts\activate.bat"

%*
