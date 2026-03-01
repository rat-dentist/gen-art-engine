$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
    & python -c "import importlib.util,sys;mods=('pyglet','numpy');missing=[m for m in mods if importlib.util.find_spec(m) is None];sys.exit(0 if not missing else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing 3D dependencies from requirements-3d.txt..."
        & python -m pip install -r requirements-3d.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Dependency installation failed. Fix the pip error above, then run Ctrl+Shift+B again."
            exit $LASTEXITCODE
        }
    }

    & python src/main_3d.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
