<#
    Create Desktop and Start Menu shortcuts for PSE-A.

        powershell -ExecutionPolicy Bypass -File build\install_shortcuts.ps1

    Removes them again with:

        powershell -ExecutionPolicy Bypass -File build\install_shortcuts.ps1 -Remove

    Everything is per-user: nothing is written outside your own profile, no
    administrator rights are needed, and nothing touches the registry.
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    # Which build to point at. The folder build starts in well under a second;
    # the single file unpacks itself to temp on every launch.
    [ValidateSet('folder', 'single')]
    [string]$Target = 'folder'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$exe = if ($Target -eq 'single') {
    Join-Path $root 'dist\PSE-A-single.exe'
} else {
    Join-Path $root 'dist\PSE-A\PSE-A.exe'
}

$desktop = [Environment]::GetFolderPath('Desktop')
$startMenu = Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'
$links = @(
    (Join-Path $desktop   'PSE Analyzer.lnk'),
    (Join-Path $startMenu 'PSE Analyzer.lnk')
)

if ($Remove) {
    foreach ($l in $links) {
        if (Test-Path -LiteralPath $l) {
            Remove-Item -LiteralPath $l -Force
            Write-Host "removed  $l"
        } else {
            Write-Host "not there $l"
        }
    }
    return
}

if (-not (Test-Path -LiteralPath $exe)) {
    throw "Executable not found: $exe`nBuild it first:  .venv\Scripts\pyinstaller.exe --noconfirm build\pse-a.spec"
}

$icon = Join-Path $PSScriptRoot 'pse-a.ico'
$shell = New-Object -ComObject WScript.Shell

foreach ($l in $links) {
    $dir = Split-Path -Parent $l
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $sc = $shell.CreateShortcut($l)
    $sc.TargetPath = $exe
    # No arguments: the launcher opens the web view when started with none.
    $sc.Arguments = ''
    $sc.WorkingDirectory = Split-Path -Parent $exe
    $sc.Description = 'PSE Analyzer - spectral analysis of local language models'
    if (Test-Path -LiteralPath $icon) { $sc.IconLocation = $icon }
    $sc.Save()
    Write-Host "created  $l"
}

Write-Host ''
Write-Host "Both shortcuts point at: $exe"
Write-Host 'Double-clicking opens the web view at http://127.0.0.1:8765'
