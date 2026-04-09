param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $projectRoot "config.json"
$pyprojectPath = Join-Path $projectRoot "pyproject.toml"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "未找到 config.json。"
}

if (-not (Test-Path -LiteralPath $pyprojectPath)) {
    throw "未找到 pyproject.toml。"
}

$config = Get-Content $configPath -Raw | ConvertFrom-Json

if (-not $config.torch_whl) {
    throw "config.json 中未配置 torch_whl。"
}

$wheelPath = [string]$config.torch_whl
$wheelFullPath = Join-Path $projectRoot $wheelPath

if (-not (Test-Path -LiteralPath $wheelFullPath)) {
    throw "未找到 torch wheel：$wheelFullPath"
}

$content = Get-Content $pyprojectPath -Raw
$content = [regex]::Replace(
    $content,
    '(?ms)\n\[\[tool\.uv\.index\]\].*?(?=\n\[tool\.uv\.sources\])',
    "`n"
)
$content = [regex]::Replace(
    $content,
    '(?m)^torch = \{.*\}$',
    ('torch = { path = "' + $wheelPath.Replace('\', '/') + '" }')
)

Set-Content -LiteralPath $pyprojectPath -Value $content -NoNewline
Write-Host "Updated pyproject.toml to use local torch wheel:" $wheelPath
