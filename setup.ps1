#Requires -Version 7.0
<#
.SYNOPSIS
    Native setup for the GraphRAG Knowledge Base Assistant (no Docker, no DB server).

.DESCRIPTION
    The assistant uses SurrealDB's embedded SurrealKV engine — the database is a
    local file opened in-process, so there is no server to install or run.

    1. Creates a Python virtual environment and installs all dependencies
    2. Patches LightRAG to register the SurrealDB storage adapter
    3. Creates .env from .env.example if one does not already exist
    4. Updates config.yaml with native paths
    5. Prints the commands needed to ingest and start the API

.NOTES
    Run from the project root directory.
    Requires Python 3.11 or later on PATH.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Constants ─────────────────────────────────────────────────────────────────

$VENV_DIR          = ".venv"
$LIGHTRAG_DATA_DIR = "lightrag_data"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-OK   { param([string]$Text) Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "    --  $Text" -ForegroundColor Yellow }
function Fail       { param([string]$Text) Write-Host "`nERROR: $Text" -ForegroundColor Red; exit 1 }

# ── 0. Preflight ──────────────────────────────────────────────────────────────

Write-Host "`nGraphRAG Assistant — Native Setup (embedded SurrealKV)" -ForegroundColor White
Write-Host   "=====================================================`n" -ForegroundColor White

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

# ── 1. Create data directory ──────────────────────────────────────────────────

Write-Step "Creating data directory"

if (-not (Test-Path $LIGHTRAG_DATA_DIR)) {
    New-Item -ItemType Directory -Path $LIGHTRAG_DATA_DIR | Out-Null
    Write-OK "Created $LIGHTRAG_DATA_DIR\  (embedded database file lives here)"
} else {
    Write-Warn "$LIGHTRAG_DATA_DIR\ already exists — skipping."
}

# ── 2. Python virtual environment ─────────────────────────────────────────────

Write-Step "Python virtual environment"

if (Test-Path "$VENV_DIR\Scripts\python.exe") {
    Write-Warn "$VENV_DIR already exists — skipping creation."
} else {
    Write-Host "    Creating $VENV_DIR\ ..."
    & $python -m venv $VENV_DIR
    Write-OK "Virtual environment created."
}

$pip     = "$VENV_DIR\Scripts\pip.exe"
$vPython = "$VENV_DIR\Scripts\python.exe"

Write-Host "    Installing dependencies from requirements.txt (this may take a few minutes)..."
& $pip install --quiet --upgrade pip
& $pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Check requirements.txt and your network connection." }
Write-OK "Dependencies installed."

# ── 3. Patch LightRAG ─────────────────────────────────────────────────────────

Write-Step "Patching LightRAG to register SurrealDB adapter"

& $vPython patch_lightrag.py
if ($LASTEXITCODE -ne 0) { Fail "patch_lightrag.py failed. See output above." }

# ── 4. Environment file ───────────────────────────────────────────────────────

Write-Step "Environment configuration (.env)"

if (Test-Path ".env") {
    Write-Warn ".env already exists — skipping. Verify LIBRARY_PATH is set correctly."
} else {
    Copy-Item ".env.example" ".env"
    Write-OK "Created .env from .env.example."
    Write-Warn "Open .env and set LIBRARY_PATH to the absolute path of your ebook library."
}

# ── 5. Update config.yaml for native paths ────────────────────────────────────

Write-Step "Updating config.yaml for native paths"

$configPath = "config.yaml"
$config     = Get-Content $configPath -Raw

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

# ── 6. Done — print run instructions ─────────────────────────────────────────

Write-Host @"

==================================
  Setup complete. Next steps:
==================================

There is no database server to start — SurrealKV is embedded and the database
file is created automatically under $LIGHTRAG_DATA_DIR\ on first run.

1. Edit .env and set LIBRARY_PATH to your ebook folder:
     LIBRARY_PATH=C:\path\to\your\books

2. Make sure Ollama is running and the model is pulled:
     ollama pull qwen2.5:14b

3. Run ingestion (one-time; re-run safely if interrupted, or add --reset to rebuild):

     .venv\Scripts\python.exe ingest.py --config config.yaml

4. Start the API (keep it running; GraphNotes / OpenWebUI connect here):

     .venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000

   Clients can then connect to: http://localhost:8000

"@ -ForegroundColor White
