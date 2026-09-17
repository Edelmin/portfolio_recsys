# ============================================================
# Wrapper para ejecutar comandos con cachés en D:\
# Uso:
#   .\run pr-run-all
#   .\run kedro run --pipeline fetch_stock_prices
#   .\run kedro viz
#   .\run pytest
#   .\run ruff check .
# ============================================================

$env:UV_CACHE_DIR = "D:\cache\uv"
$env:PIP_CACHE_DIR = "D:\cache\pip"
$env:TORCH_HOME = "D:\cache\torch"
$env:RUFF_CACHE_DIR = "D:\cache\ruff"

# Activar el entorno virtual si no está activo
if (-not $env:VIRTUAL_ENV) {
    & "$PSScriptRoot\.venv\Scripts\Activate.ps1"
}

# Ejecutar el comando pasado como argumento
if ($args.Length -le 1) {
    & $args[0]
} else {
    & $args[0] @($args[1..($args.Length - 1)])
}
