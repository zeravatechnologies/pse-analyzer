# Build the Windows release files into dist\release:
#   PSE-Analyzer-<version>-windows-x64.exe          the single file (unpacks to temp on every start)
#   PSE-Analyzer-<version>-windows-x64-folder.zip   the folder build (starts in half a second)
#   SHA256SUMS.txt
# Run from the repository root in a Python environment that has src\requirements.txt installed:
#   powershell -ExecutionPolicy Bypass -File build\windows.ps1
# torch is deliberately not bundled (see src\pse_a\shell\torch_env.py).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Version = python -c "import sys; sys.path.insert(0, 'src'); from pse_a import __version__; print(__version__)"
$Exe = "PSE-Analyzer-$Version-windows-x64.exe"
$Zip = "PSE-Analyzer-$Version-windows-x64-folder.zip"
Write-Host "== PSE Analyzer $Version -> $Exe, $Zip"

pip install -q pyinstaller pip-licenses
Remove-Item -Recurse -Force build\pyi, dist\PSE-A -ErrorAction SilentlyContinue
Remove-Item -Force dist\PSE-A-single.exe -ErrorAction SilentlyContinue
pyinstaller --noconfirm --workpath build\pyi --distpath dist build\pse-a.spec | Out-Null
if (-not (Test-Path dist\PSE-A\PSE-A.exe)) { throw "PyInstaller produced no folder build" }
if (-not (Test-Path dist\PSE-A-single.exe)) { throw "PyInstaller produced no single file" }

# papers that travel with the folder build
(Get-Content build\README-FIRST.txt -Raw).Replace("__VERSION__", $Version) |
    Set-Content dist\PSE-A\README-FIRST.txt -NoNewline
if (Test-Path LICENSE.md) { Copy-Item LICENSE.md dist\PSE-A\LICENSE.md }
if (Test-Path NOTICE) { Copy-Item NOTICE dist\PSE-A\NOTICE.txt }
pip-licenses --format=plain-vertical --with-license-file --no-license-path --with-urls `
    --ignore-packages pip setuptools wheel pip-licenses pyinstaller pyinstaller-hooks-contrib `
    | Set-Content dist\PSE-A\THIRD-PARTY-NOTICES.txt

New-Item -ItemType Directory -Force dist\release | Out-Null
Copy-Item dist\PSE-A-single.exe dist\release\$Exe -Force
Remove-Item dist\release\$Zip -ErrorAction SilentlyContinue
Compress-Archive -Path dist\PSE-A -DestinationPath dist\release\$Zip -CompressionLevel Optimal
# the same papers beside the single file, for those who take only the .exe
Copy-Item dist\PSE-A\README-FIRST.txt dist\release\README-FIRST.txt -Force
Copy-Item dist\PSE-A\THIRD-PARTY-NOTICES.txt dist\release\THIRD-PARTY-NOTICES.txt -Force
foreach ($f in @($Exe, $Zip)) {
    $Hash = (Get-FileHash dist\release\$f -Algorithm SHA256).Hash.ToLower()
    [IO.File]::AppendAllText("$Root\dist\release\SHA256SUMS.txt", "$Hash  $f`n")   # LF, so sha256sum -c reads it
}
Get-ChildItem dist\release | Format-Table Name, @{n="MB";e={[math]::Round($_.Length/1MB,1)}}
Write-Host "WINDOWS DONE"
