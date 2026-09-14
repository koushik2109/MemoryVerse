param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$RunBackend = $true
$RunFrontend = $true

if ($BackendOnly) { $RunFrontend = $false }
if ($FrontendOnly) { $RunBackend = $false }
if ($BackendOnly -and $FrontendOnly) {
    $RunBackend = $true
    $RunFrontend = $true
}

$ScriptDir = $PSScriptRoot
$BackendDir = Join-Path $ScriptDir "backend"
$FrontendDir = Join-Path $ScriptDir "frontend"
$VenvDir = Join-Path $ScriptDir ".venv"

function Write-Ok ($msg) { Write-Host "[OK]  $msg" -ForegroundColor Green }
function Write-Warn ($msg) { Write-Host "[!]   $msg" -ForegroundColor Yellow }
function Write-Err ($msg) { Write-Host "[ERR] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "========================================"
Write-Host "       MemoryVerse Dev Launcher         "
Write-Host "========================================"
Write-Host ""

if ($RunBackend) {
    if (-not (Test-Path $BackendDir)) { Write-Err "Backend directory not found: $BackendDir" }

    Write-Warn "Checking for existing backend processes..."
    $pids = @()
    try {
        $connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        }
    } catch {}

    if ($pids.Count -gt 0) {
        Write-Warn "Killing existing process on port 8000 (PID: $($pids -join ', '))..."
        foreach ($p in $pids) {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            cmd /c "taskkill /F /PID $p" 2>$null
        }
        Start-Sleep -Seconds 1
    }

    $FastapiCheck = Join-Path $VenvDir "Lib\site-packages\fastapi"
    $ExifreadCheck = Join-Path $VenvDir "Lib\site-packages\exifread"
    if ((-not (Test-Path $VenvDir)) -or (-not (Test-Path $FastapiCheck)) -or (-not (Test-Path $ExifreadCheck))) {
        Write-Warn "Virtual environment dependencies missing or incomplete at $VenvDir -- installing..."
        $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
        if (-not (Test-Path $VenvDir)) {
            python -m venv $VenvDir
            & $PythonExe -m ensurepip
        }
        & $PythonExe -m pip install -q --upgrade pip
        & $PythonExe -m pip install -q -r (Join-Path $BackendDir "requirements.txt")
        Write-Ok "Virtual environment created and dependencies installed."
    } else {
        Write-Ok "Virtual environment found."
    }
}

if ($RunFrontend) {
    if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
        Write-Err "Flutter is not in PATH. Install from https://flutter.dev"
    }
    
    try {
        $flutterVer = (flutter --version | Select-Object -First 1)
        Write-Ok "Flutter found: $flutterVer"
    } catch {
        Write-Ok "Flutter found."
    }
}

# Launching terminals
if ($RunBackend) {
    Write-Ok "Opening Backend terminal..."
    $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
    $BackendCmd = "cd '$BackendDir'; `$env:PYTHONPATH='.'; Write-Host ''; Write-Host '  >>> MemoryVerse Backend starting (Uvicorn with Auto-Reload)...'; Write-Host ''; & '$PythonExe' -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000"
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $BackendCmd
}

if ($RunBackend -and $RunFrontend) {
    Start-Sleep -Seconds 1
}

if ($RunFrontend) {
    Write-Ok "Starting Frontend in this terminal (press 'r' for Hot Reload, 'R' for Hot Restart)..."
    Write-Host ""
    Write-Host "  API Backend -> http://localhost:8000"
    Write-Host "  API Docs    -> http://localhost:8000/docs"
    Write-Host ""
    Set-Location $FrontendDir
    flutter run
}

Write-Host ""
Write-Host "  Press Ctrl+C in each terminal window to stop."
Write-Host ""
