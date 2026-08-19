@echo off
REM A simple wrapper to launch dev.ps1 for convenience
powershell.exe -ExecutionPolicy Bypass -File "%~dp0dev.ps1" %*
