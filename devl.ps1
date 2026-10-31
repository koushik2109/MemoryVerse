[CmdletBinding()]
param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [switch]$SkipAiEngine,
    [switch]$PrepareFlutter,
    [switch]$Analyze,
    [string]$DeviceId,
    [int]$HealthTimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'

if ($BackendOnly -and $FrontendOnly) {
    throw 'Choose either -BackendOnly or -FrontendOnly, not both.'
}

$script:RepoRoot = $PSScriptRoot
$script:BackendDir = Join-Path $script:RepoRoot 'backend'
$script:FrontendDir = Join-Path $script:RepoRoot 'frontend'
$script:PythonExe = Join-Path $script:RepoRoot '.venv\Scripts\python.exe'
$script:AiPythonExe = Join-Path $script:RepoRoot 'ai_engine\rag\.venv\Scripts\python.exe'
$script:BindHost = if ([string]::IsNullOrWhiteSpace($env:MEMORYVERSE_BIND_HOST)) { '127.0.0.1' } else { $env:MEMORYVERSE_BIND_HOST.Trim() }
if (-not (Test-Path -LiteralPath $script:AiPythonExe)) {
    $script:AiPythonExe = $script:PythonExe
}

function Write-Ok([string]$Message) {
    Write-Host "[OK]  $Message" -ForegroundColor Green
}

function Write-Info([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
    Write-Host "[!]   $Message" -ForegroundColor Yellow
}

function ConvertTo-PowerShellLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function Test-ListeningPort([int]$Port) {
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1)
    } catch {
        return $false
    }
}

function Test-HealthyService([string]$Url) {
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec 2
        return $response.status -eq 'healthy'
    } catch {
        return $false
    }
}

function Wait-ForHealthyService([string]$Name, [string]$Url, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HealthyService $Url) {
            Write-Ok "$Name is healthy at $Url"
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become healthy within $TimeoutSeconds seconds. Check its terminal window."
}

$shellCommand = if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh.exe' } else { 'powershell.exe' }
$shell = Get-Command $shellCommand -ErrorAction SilentlyContinue
if (-not $shell) { throw "Could not locate $shellCommand to open service terminals." }
$shellPath = $shell.Source

$runBackend = -not $FrontendOnly
$runFrontend = -not $BackendOnly
$runAi = $runBackend -and -not $SkipAiEngine

Write-Host ''
Write-Host '========================================'
Write-Host ' MemoryVerse Windows Dev Launcher (devl)'
Write-Host '========================================'

if ($runBackend -and -not (Test-Path -LiteralPath $script:BackendDir)) {
    throw "Backend folder not found: $script:BackendDir"
}
if ($runFrontend -and -not (Test-Path -LiteralPath $script:FrontendDir)) {
    throw "Frontend folder not found: $script:FrontendDir"
}
if (($runBackend -or $runAi) -and -not (Test-Path -LiteralPath $script:PythonExe)) {
    throw "Project Python environment is missing: $script:PythonExe. Set up .venv and install the approved requirements before running devl.ps1."
}
if ($runBackend) {
    & $script:PythonExe -c 'import fastapi, uvicorn' 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "The root .venv is missing FastAPI or Uvicorn. Install backend/requirements.txt into $script:PythonExe's environment first."
    }
}
if ($runAi) {
    $aiProbe = @'
import importlib.util
modules = ("fastapi", "uvicorn", "cv2", "transformers", "torch", "librosa", "whisper", "langgraph", "FlagEmbedding")
missing = [name for name in modules if importlib.util.find_spec(name) is None]
print("Missing AI RAG modules: " + ", ".join(missing) if missing else "AI RAG modules are available.")
raise SystemExit(bool(missing))
'@
    & $script:AiPythonExe -c $aiProbe
    if ($LASTEXITCODE -ne 0) {
        throw "AI RAG dependencies are incomplete. Install ai_engine/rag/requirements.txt into $script:AiPythonExe's environment first."
    }
}

if ($runFrontend) {
    $flutter = Get-Command flutter -ErrorAction SilentlyContinue
    if (-not $flutter) { throw 'Flutter is not in PATH.' }

    $requestedDevice = if ($DeviceId) { $DeviceId } else { $env:FLUTTER_DEVICE }
    try {
        $flutterDevices = @(flutter devices --machine | ConvertFrom-Json)
    } catch {
        throw "Could not read Flutter devices. Start a device or simulator, then retry. $($_.Exception.Message)"
    }
    $availableDevices = @($flutterDevices | Where-Object { $_.id })

    if ($requestedDevice) {
        $selectedDevice = $availableDevices | Where-Object { $_.id -eq $requestedDevice } | Select-Object -First 1
        if (-not $selectedDevice) {
            $available = if ($availableDevices.Count) { $availableDevices.id -join ', ' } else { 'none' }
            throw "Flutter device '$requestedDevice' is not ready. Available devices: $available."
        }
        $DeviceId = $selectedDevice.id
    } elseif ($availableDevices.Count -eq 1) {
        $DeviceId = $availableDevices[0].id
    } elseif ($availableDevices.Count -gt 1) {
        throw "Multiple Flutter devices are ready ($($availableDevices.id -join ', ')). Choose one with -DeviceId."
    } else {
        throw 'No running Flutter device was found. Start a device or simulator, then run devl.ps1 again.'
    }

    if (-not $PrepareFlutter -and -not (Test-Path -LiteralPath (Join-Path $script:FrontendDir '.dart_tool\package_config.json'))) {
        throw 'Flutter packages are not prepared. Run `flutter pub get` in frontend, or rerun devl.ps1 with -PrepareFlutter.'
    }
    Write-Ok "Flutter will target device $DeviceId"
}

function Start-ServiceTerminal([string]$Title, [string]$Code) {
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($Code))
    Start-Process -FilePath $shellPath `
        -ArgumentList @('-NoLogo', '-NoExit', '-EncodedCommand', $encoded) `
        -WindowStyle Normal | Out-Null
    Write-Ok "Opened $Title terminal."
}

function Start-PythonService([string]$Name, [string]$WorkingDirectory, [string]$Python, [string]$Module, [int]$Port, [string]$ReloadDir, [string]$BindHost = '127.0.0.1') {
    if (Test-ListeningPort $Port) {
        if (Test-HealthyService "http://127.0.0.1:$Port/health") {
            Write-Ok "$Name is already healthy on port $Port; keeping it running."
            return
        }
        throw "Port $Port is already in use by a service that did not pass its health check. Stop or inspect that service, then retry."
    }

    $repoLiteral = ConvertTo-PowerShellLiteral $script:RepoRoot
    $workingLiteral = ConvertTo-PowerShellLiteral $WorkingDirectory
    $pythonLiteral = ConvertTo-PowerShellLiteral $Python
    $hostLiteral = ConvertTo-PowerShellLiteral $BindHost
    $titleLiteral = ConvertTo-PowerShellLiteral "MemoryVerse $Name"
    $code = @"
`$Host.UI.RawUI.WindowTitle = $titleLiteral
Set-Location -LiteralPath $workingLiteral
`$env:PYTHONPATH = $repoLiteral
Write-Host 'Starting $Name on port $Port...' -ForegroundColor Cyan
& $pythonLiteral -m uvicorn $Module --reload --reload-dir $ReloadDir --host $hostLiteral --port $Port
if (`$LASTEXITCODE -ne 0) { Write-Host '$Name exited with code ' `$LASTEXITCODE -ForegroundColor Red }
Write-Host 'This window contains the $Name output. Close it when you are done.' -ForegroundColor Yellow
"@
    Start-ServiceTerminal $Name $code
}

if ($runAi) {
    Start-PythonService 'AI RAG Engine' $script:RepoRoot $script:AiPythonExe 'ai_engine.rag.server:app' 8001 'ai_engine'
}
if ($runBackend) {
    Start-PythonService 'Backend' $script:BackendDir $script:PythonExe 'app.main:app' 8000 'app' $script:BindHost
}

if ($runBackend) {
    Wait-ForHealthyService 'Backend' 'http://127.0.0.1:8000/health' $HealthTimeoutSeconds
}
if ($runAi) {
    Wait-ForHealthyService 'AI RAG Engine' 'http://127.0.0.1:8001/health' $HealthTimeoutSeconds
}

if ($runFrontend) {
    $frontendLiteral = ConvertTo-PowerShellLiteral $script:FrontendDir
    $deviceLiteral = ConvertTo-PowerShellLiteral $DeviceId
    $titleLiteral = ConvertTo-PowerShellLiteral 'MemoryVerse Flutter App'
    $analyzeLine = if ($Analyze) { "flutter analyze`nif (`$LASTEXITCODE -ne 0) { throw 'Flutter analysis failed; app launch stopped.' }" } else { "Write-Host 'Skipping flutter analyze for faster startup. Use -Analyze to include it.' -ForegroundColor DarkGray" }
    $pubGetLine = if ($PrepareFlutter) { "flutter pub get`nif (`$LASTEXITCODE -ne 0) { throw 'flutter pub get failed.' }" } else { "Write-Host 'Using prepared Flutter dependencies. Pass -PrepareFlutter to refresh them.' -ForegroundColor DarkGray" }
    $frontendCode = @"
`$Host.UI.RawUI.WindowTitle = $titleLiteral
Set-Location -LiteralPath $frontendLiteral
Write-Host 'Preparing Flutter dependencies...' -ForegroundColor Cyan
$pubGetLine
$analyzeLine
Write-Host 'Launching MemoryVerse on Flutter device $DeviceId...' -ForegroundColor Cyan
flutter run -d $deviceLiteral
if (`$LASTEXITCODE -ne 0) { Write-Host 'Flutter exited with code ' `$LASTEXITCODE -ForegroundColor Red }
Write-Host 'This window contains the Flutter output. Close it when you are done.' -ForegroundColor Yellow
"@
    Start-ServiceTerminal 'Flutter Frontend' $frontendCode
}

Write-Host ''
Write-Host 'Services are running in their own output windows.' -ForegroundColor Green
Write-Host 'Use Ctrl+C in a service window to stop that service.' -ForegroundColor Green
