# Starts the AMMBA PoC stack locally without Docker (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
#
# Opens four minimized console windows (close them or run stop-local.ps1 to
# stop). All data is in-memory: starting fresh always means an empty system.

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"

# First run: create the venv and install dependencies.
# (execution-node requirements cover every service for mock mode;
#  web3 is only needed for BLOCKCHAIN_MODE=live, see README.)
if (-not (Test-Path $py)) {
    Write-Host "First run: creating .venv and installing dependencies..."
    python -m venv (Join-Path $root ".venv")
    & $py -m pip install --disable-pip-version-check --timeout 90 `
        -r (Join-Path $root "amm-execution-node\requirements.txt")
}

$services = @(
    @{ Name = "mock-offchain-db";   Dir = "mock-offchain-db";   Port = 8080; Args = @("-m","uvicorn","src.main:app","--port","8080") },
    @{ Name = "amm-clearing-node";  Dir = "amm-clearing-node";  Port = 8081; Args = @("-m","uvicorn","src.main:app","--port","8081") },
    @{ Name = "amm-execution-node"; Dir = "amm-execution-node"; Port = 8082; Args = @("-m","uvicorn","src.main:app","--port","8082") },
    @{ Name = "ui (static server)"; Dir = ".";                  Port = 3000; Args = @("-m","http.server","3000","--directory","ui") }
)

foreach ($svc in $services) {
    $busy = Get-NetTCPConnection -LocalPort $svc.Port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        Write-Host "  port $($svc.Port) already in use - skipping $($svc.Name) (run stop-local.ps1 first for a clean restart)"
        continue
    }
    Start-Process -FilePath $py -ArgumentList $svc.Args `
        -WorkingDirectory (Join-Path $root $svc.Dir) -WindowStyle Minimized
    Write-Host "  started $($svc.Name) on port $($svc.Port)"
}

# Wait for health endpoints — shared 90s budget; cold starts can be slow
# when all four processes launch at once (imports + antivirus scanning).
# NOTE: probe 127.0.0.1, not "localhost" — uvicorn binds IPv4 only, and on
# some Windows setups "localhost" resolves to ::1 first and hangs in
# PowerShell instead of falling back (browsers are unaffected).
$pending = @(
    @{ Url = "http://127.0.0.1:8080/health_check"; Name = "off-chain DB" },
    @{ Url = "http://127.0.0.1:8081/health";       Name = "clearing node" },
    @{ Url = "http://127.0.0.1:8082/health";       Name = "execution node" }
)
Write-Host "Waiting for services (up to 90s)..."
$deadline = (Get-Date).AddSeconds(90)
while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
    $still = @()
    foreach ($check in $pending) {
        try {
            Invoke-RestMethod $check.Url -TimeoutSec 2 | Out-Null
            Write-Host "  OK   $($check.Name)"
        } catch { $still += $check }
    }
    $pending = $still
    if ($pending.Count -gt 0) { Start-Sleep -Milliseconds 700 }
}
foreach ($check in $pending) {
    Write-Host "  FAIL $($check.Name) not reachable after 90s - check its console window"
}

Write-Host ""
Write-Host "AMMBA stack is running:"
Write-Host "  UI              http://localhost:3000"
Write-Host "  off-chain DB    http://localhost:8080"
Write-Host "  clearing node   http://localhost:8081"
Write-Host "  execution node  http://localhost:8082"
Write-Host ""
Write-Host "Stop everything:        scripts\stop-local.ps1"
Write-Host "Wipe data (no restart): curl.exe -X POST http://127.0.0.1:8080/reset"
