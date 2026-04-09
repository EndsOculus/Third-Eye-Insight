param(
    [switch]$EnsureLocalTorch
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$localTorchScript = Join-Path $projectRoot "use_local_torch_wheel.ps1"
$trainScript = Join-Path $projectRoot "train.py"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "未找到 .venv\\Scripts\\python.exe。请先运行 uv sync 创建虚拟环境。"
}

if (-not (Test-Path -LiteralPath $trainScript)) {
    throw "未找到 train.py。"
}

if ($EnsureLocalTorch -and (Test-Path -LiteralPath $localTorchScript)) {
    & $localTorchScript
    if ($LASTEXITCODE -ne 0) {
        throw "本地 torch 预安装失败。"
    }
}

Write-Host "Running train.py with the existing .venv environment..."
& $venvPython $trainScript

exit $LASTEXITCODE
