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
        }
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
    $BackendCmd = "Set-Location '$BackendDir'; & '$VenvDir\Scripts\Activate.ps1'; Write-Host ''; Write-Host '  >>> MemoryVerse Backend starting...'; Write-Host ''; `$env:PYTHONPATH='.'; python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $BackendCmd
}

if ($RunBackend -and $RunFrontend) {
    Start-Sleep -Seconds 1
}

if ($RunFrontend) {
    Write-Ok "Opening Frontend terminal..."
    $FrontendCmd = "cd '$FrontendDir'; Write-Host ''; Write-Host '  >>> MemoryVerse Flutter starting...'; Write-Host ''; flutter run"
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $FrontendCmd
}

Write-Host ""
if ($RunBackend -and $RunFrontend) {
    Write-Ok "Both services launching!"
    Write-Host ""
    Write-Host "  API Backend -> http://localhost:8000"
    Write-Host "  API Docs    -> http://localhost:8000/docs"
    Write-Host "  Frontend    -> check the Flutter terminal for device selection"
} elseif ($RunBackend) {
    Write-Host "  API Backend -> http://localhost:8000"
} elseif ($RunFrontend) {
    Write-Host "  Frontend    -> check the Flutter terminal"
}

Write-Host ""
Write-Host "  Press Ctrl+C in each terminal window to stop."
Write-Host ""
