[CmdletBinding()]
param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [string]$DeviceId
)

# Compatibility entry point. devl.ps1 is the maintained Windows launcher.
$ErrorActionPreference = 'Stop'
$arguments = @{}
if ($BackendOnly) { $arguments.BackendOnly = $true }
if ($FrontendOnly) { $arguments.FrontendOnly = $true }
if ($DeviceId) { $arguments.DeviceId = $DeviceId }
& (Join-Path $PSScriptRoot 'devl.ps1') @arguments
