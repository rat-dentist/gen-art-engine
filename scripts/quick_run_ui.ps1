$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
    $pythonCmd = $null
    $pythonPrefix = @()

    if (Test-Path ".venv\Scripts\python.exe") {
        $pythonCmd = (Resolve-Path ".venv\Scripts\python.exe").Path
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = "py"
        $pythonPrefix = @("-3")
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = "python"
    }
    else {
        Write-Host "Python was not found. Create .venv or install Python, then run Ctrl+Shift+B again."
        exit 1
    }

    & $pythonCmd @pythonPrefix -c "import importlib.util,sys;mods=('PySide6','numpy','PIL','cv2');missing=[m for m in mods if importlib.util.find_spec(m) is None];sys.exit(0 if not missing else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing UI dependencies from requirements.txt..."
        & $pythonCmd @pythonPrefix -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Dependency installation failed. Fix the pip error above, then run Ctrl+Shift+B again."
            exit $LASTEXITCODE
        }
    }

    & $pythonCmd @pythonPrefix src/ui_app.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
