param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [string]$DeviceId
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

    Write-Warn "Checking backend port 8000..."
    $backendAlreadyRunning = $false
    try {
        $connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $backendAlreadyRunning = $true
        }
    } catch {}

    if ($backendAlreadyRunning) {
        Write-Warn "A service is already listening on port 8000; keeping it running."
    }

    $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $PythonExe)) {
        Write-Err "Project virtual environment is missing: $PythonExe. Create it and install backend/requirements.txt before starting the backend."
    }

    & $PythonExe -c "import fastapi, uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Err "The project .venv is missing FastAPI or Uvicorn. Install the approved backend requirements into $VenvDir, then rerun dev.ps1. No packages were installed by this launcher."
    }
    Write-Ok "Project virtual environment and backend runtime are available."
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

    try {
        $devices = @(flutter devices --machine | ConvertFrom-Json)
        $androidDevices = @($devices | Where-Object { $_.targetPlatform -like "android-*" })
        if ($DeviceId) {
            $selectedDevice = $androidDevices | Where-Object { $_.id -eq $DeviceId } | Select-Object -First 1
            if (-not $selectedDevice) {
                $availableDevices = @($androidDevices | ForEach-Object { $_.id })
                $availableList = if ($availableDevices.Count) { $availableDevices -join ", " } else { "none" }
                Write-Err "Android device '$DeviceId' is not connected. Running Android devices: $availableList. Start the emulator or choose a listed device."
            }
        } elseif ($androidDevices.Count -eq 1) {
            $DeviceId = $androidDevices[0].id
        } elseif ($androidDevices.Count -gt 1) {
            Write-Warn "Multiple Android devices are connected: $($androidDevices.id -join ', '). Choose one with -DeviceId."
        }
    } catch {
        Write-Warn "Could not read Flutter's device list; Flutter will use its normal device selection."
    }

    if ($DeviceId) {
        Write-Ok "Flutter will run on device: $DeviceId"
    } else {
        Write-Warn "No single running Android emulator was detected. Start one with emulator.ps1 or pass -DeviceId."
    }
}

# Launching terminals
if ($RunBackend) {
    if (-not $backendAlreadyRunning) {
        Write-Ok "Opening Backend terminal..."
        $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
        $BackendCmd = "Set-Location '$BackendDir'; `$env:PYTHONPATH='.'; Write-Host ''; Write-Host '  >>> MemoryVerse Backend starting (Uvicorn with Auto-Reload)...'; Write-Host ''; & '$PythonExe' -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000"
        Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $BackendCmd
    }
}

if ($RunBackend -and $RunFrontend) {
    Write-Warn "Waiting for the backend health check..."
    $healthUrl = "http://127.0.0.1:8000/health"
    $healthDeadline = (Get-Date).AddSeconds(180)
    $backendHealthy = $false

    while ((Get-Date) -lt $healthDeadline) {
        try {
            $healthResponse = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 2 -UseBasicParsing
            $healthBody = $healthResponse.Content | ConvertFrom-Json
            if ($healthResponse.StatusCode -eq 200 -and $healthBody.status -eq "healthy") {
                $backendHealthy = $true
                break
            }
        } catch {}

        Start-Sleep -Milliseconds 500
    }

    if (-not $backendHealthy) {
        Write-Err "Backend did not become healthy within 180 seconds. Check the backend terminal before starting the app."
    }
    Write-Ok "Backend health check passed."
}

if ($RunFrontend) {
    Write-Ok "Starting Frontend in this terminal (press 'r' for Hot Reload, 'R' for Hot Restart)..."
    Write-Host ""
    Write-Host "  API Backend -> http://localhost:8000"
    Write-Host "  API Docs    -> http://localhost:8000/docs"
    Write-Host ""
    Set-Location $FrontendDir
    if ($DeviceId) {
        flutter run -d $DeviceId
    } else {
        flutter run
    }
}

Write-Host ""
Write-Host "  Press Ctrl+C in each terminal window to stop."
Write-Host ""
