#Requires -Version 5.1
<#
.SYNOPSIS
    Native setup for GraphRAG Knowledge Base Assistant (no Docker required).

.DESCRIPTION
    1. Downloads the SurrealDB v2.1.4 binary for Windows
    2. Creates a Python virtual environment and installs all dependencies
    3. Patches LightRAG to register the SurrealDB storage adapter
    4. Creates .env from .env.example if one does not already exist
    5. Updates config.yaml with native (non-container) paths
    6. Prints the commands needed to start SurrealDB and the API

.NOTES
    Run from the project root directory.
    Requires Python 3.11 or later on PATH.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Constants ─────────────────────────────────────────────────────────────────

$SURREALDB_VERSION = "2.1.4"
$SURREALDB_EXE     = "surreal.exe"
$SURREALDB_URL     = "https://github.com/surrealdb/surrealdb/releases/download/v$SURREALDB_VERSION/surreal-v$SURREALDB_VERSION.windows-amd64.exe"
$VENV_DIR          = ".venv"
$DATA_DIR          = "surrealdb_data"
$LIGHTRAG_DATA_DIR = "lightrag_data"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-OK   { param([string]$Text) Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "    --  $Text" -ForegroundColor Yellow }
function Fail       { param([string]$Text) Write-Host "`nERROR: $Text" -ForegroundColor Red; exit 1 }

# ── 0. Preflight ──────────────────────────────────────────────────────────────

Write-Host "`nGraphRAG Assistant — Native Setup" -ForegroundColor White
Write-Host   "==================================`n" -ForegroundColor White

Write-Step "Checking prerequisites"

if (-not (Test-Path "requirements.txt")) {
    Fail "Run this script from the project root directory (where requirements.txt lives)."
}

$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -lt 11) {
                Fail "Python 3.11+ required, found: $ver"
            }
            $python = $candidate
            Write-OK "Found $ver"
            break
        }
    } catch { }
}
if (-not $python) { Fail "Python 3.11+ not found on PATH. Install from https://www.python.org/downloads/" }

# ── 1. Download SurrealDB ─────────────────────────────────────────────────────

Write-Step "SurrealDB binary (v$SURREALDB_VERSION)"

if (Test-Path $SURREALDB_EXE) {
    Write-Warn "$SURREALDB_EXE already exists — skipping download."
} else {
    Write-Host "    Downloading from GitHub releases..." -NoNewline
    try {
        # Use HttpClient for better progress and reliability than Invoke-WebRequest
        Add-Type -AssemblyName System.Net.Http
        $handler  = [System.Net.Http.HttpClientHandler]::new()
        $handler.AllowAutoRedirect = $true
        $client   = [System.Net.Http.HttpClient]::new($handler)
        $client.Timeout = [TimeSpan]::FromMinutes(5)

        $response = $client.GetAsync($SURREALDB_URL).GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            $client.Dispose()
            Fail "Download failed: HTTP $([int]$response.StatusCode). Check the URL or download manually:`n    $SURREALDB_URL"
        }
        $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        [System.IO.File]::WriteAllBytes((Join-Path $PWD $SURREALDB_EXE), $bytes)
        $client.Dispose()
        Write-Host " done" -ForegroundColor Green
        Write-OK "Saved as $SURREALDB_EXE ($([math]::Round($bytes.Length / 1MB, 1)) MB)"
    } catch {
        Fail "Download error: $_`n`n    You can download manually from:`n    $SURREALDB_URL`n    and place it in this directory as surreal.exe"
    }
}

# ── 2. Create data directories ────────────────────────────────────────────────

Write-Step "Creating data directories"

foreach ($dir in @($DATA_DIR, $LIGHTRAG_DATA_DIR)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-OK "Created $dir\"
    } else {
        Write-Warn "$dir\ already exists — skipping."
    }
}

# ── 3. Python virtual environment ─────────────────────────────────────────────

Write-Step "Python virtual environment"

if (Test-Path "$VENV_DIR\Scripts\python.exe") {
    Write-Warn "$VENV_DIR already exists — skipping creation."
} else {
    Write-Host "    Creating $VENV_DIR\ ..."
    & $python -m venv $VENV_DIR
    Write-OK "Virtual environment created."
}

$pip    = "$VENV_DIR\Scripts\pip.exe"
$vPython = "$VENV_DIR\Scripts\python.exe"

Write-Host "    Installing dependencies from requirements.txt (this may take a few minutes)..."
& $pip install --quiet --upgrade pip
& $pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Check requirements.txt and your network connection." }
Write-OK "Dependencies installed."

# ── 4. Patch LightRAG ─────────────────────────────────────────────────────────

Write-Step "Patching LightRAG to register SurrealDB adapter"

& $vPython patch_lightrag.py
if ($LASTEXITCODE -ne 0) { Fail "patch_lightrag.py failed. See output above." }

# ── 5. Environment file ───────────────────────────────────────────────────────

Write-Step "Environment configuration (.env)"

if (Test-Path ".env") {
    Write-Warn ".env already exists — skipping. Verify LIBRARY_PATH is set correctly."
} else {
    Copy-Item ".env.example" ".env"
    Write-OK "Created .env from .env.example."
    Write-Warn "Open .env and set LIBRARY_PATH to the absolute path of your ebook library."
}

# ── 6. Update config.yaml for native paths ────────────────────────────────────

Write-Step "Updating config.yaml for native paths"

$configPath = "config.yaml"
$config     = Get-Content $configPath -Raw

# Replace container-specific paths with local equivalents
$nativeLibrary = "./library"
$nativeWorkDir = "./$LIGHTRAG_DATA_DIR"

$updated = $config `
    -replace "(?m)^(library_path:\s*).*$", "`${1}$nativeLibrary" `
    -replace "(?m)^(working_dir:\s*).*$",  "`${1}$nativeWorkDir"

if ($updated -ne $config) {
    Set-Content $configPath $updated -NoNewline
    Write-OK "Updated library_path  → $nativeLibrary"
    Write-OK "Updated working_dir   → $nativeWorkDir"
    Write-Warn "If your library is elsewhere, edit config.yaml and set LIBRARY_PATH in .env."
} else {
    Write-Warn "config.yaml already uses native paths — no changes made."
}

# ── 7. Done — print run instructions ─────────────────────────────────────────

$dbPath = (Resolve-Path $DATA_DIR -ErrorAction SilentlyContinue)?.Path ?? "$PWD\$DATA_DIR"

Write-Host @"

==================================
  Setup complete. Next steps:
==================================

1. Edit .env and set LIBRARY_PATH to your ebook folder:
     LIBRARY_PATH=C:\path\to\your\books

2. Start SurrealDB in a terminal (keep it running):

     .\surreal.exe start --log info --user root --pass root --bind 0.0.0.0:8001 rocksdb://$DATA_DIR\graphrag.db

3. Run ingestion (one-time, re-run safely if interrupted):

     .venv\Scripts\python.exe ingest.py --config config.yaml

4. Start the API (keep it running, connects to GraphNotes):

     .venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000 --reload

   GraphNotes can then connect to: http://localhost:8000

"@ -ForegroundColor White
