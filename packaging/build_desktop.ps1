# Build the Windows desktop installer:
#   1. PyInstaller -> packaging\dist\rag-backend\   (onedir backend, see backend.spec)
#   2. Tauri       -> desktop\src-tauri\target\release\bundle\nsis\*-setup.exe
#
# Prerequisites (once): .venv with requirements.txt + packaging\requirements-build.txt,
# Rust (rustup, MSVC toolchain), Visual Studio 2022 Build Tools (C++), Node.js.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_desktop.ps1 [-SkipBackend] [-SkipTests] [-BackendOnly] [-Python <path>] [-Config <json>]
#
# -Python picks the interpreter, e.g. the main checkout's .venv when building from a git
# worktree that has none of its own. -Config adds a Tauri config on top of tauri.conf.json:
# turning updates on, or the rename (packaging/README_RELEASE.md); give it once per file.
# -BackendOnly stops after PyInstaller, so the backend can be code-signed before a second run
# with -SkipBackend packs it into the installer (.github/workflows/release.yml does this).
param(
    [switch]$SkipBackend,
    [switch]$SkipTests,
    [switch]$BackendOnly,
    [string]$Python = "",
    [string[]]$Config = @()
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $Repo ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) { throw "Python not found: $Python (pass -Python <path to python.exe>)" }

function Step($name, [scriptblock]$body) {
    Write-Host "==> $name" -ForegroundColor Cyan
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $body
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("    done in {0:n1}s" -f $watch.Elapsed.TotalSeconds)
}

Push-Location $Repo
try {
    if (-not $SkipTests) {
        Step "pytest" { & $Python -m pytest project\tests -q }
    }
    if (-not $SkipBackend) {
        Step "PyInstaller backend (onedir)" {
            & $Python -m PyInstaller packaging\backend.spec --noconfirm --clean `
                --distpath packaging\dist --workpath packaging\build
        }
        # vec0.dll must be inside the bundle, or the embedded store cannot load.
        if (-not (Get-ChildItem packaging\dist\rag-backend -Recurse -Filter vec0.dll)) {
            throw "vec0.dll missing from packaging\dist\rag-backend"
        }
    }
    if ($BackendOnly) { return }
    Push-Location desktop
    try {
        Step "npm ci" { npm ci --no-fund --no-audit }
        $overlays = @($Config | ForEach-Object { "--config"; (Resolve-Path (Join-Path $Repo $_)).Path })
        Step "tauri build (NSIS)" { npx tauri build @overlays }
    } finally {
        Pop-Location
    }
    $installer = Get-ChildItem desktop\src-tauri\target\release\bundle\nsis\*-setup.exe |
        Sort-Object LastWriteTime | Select-Object -Last 1
    Write-Host ("Installer: {0} ({1:n1} MB)" -f $installer.FullName, ($installer.Length / 1MB)) -ForegroundColor Green
} finally {
    Pop-Location
}
