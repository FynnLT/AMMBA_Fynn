# Stops the locally running AMMBA PoC stack (counterpart of start-local.ps1).
#   powershell -ExecutionPolicy Bypass -File scripts\stop-local.ps1
#
# Kills the listeners on the four AMMBA ports — but only python processes,
# so an unrelated app on one of these ports is left alone.

$ports = 8080, 8081, 8082, 3000
$stopped = 0

foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in ($conns | Select-Object -Unique OwningProcess)) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($null -eq $proc) { continue }
        if ($proc.ProcessName -like "python*") {
            try {
                Stop-Process -Id $proc.Id -Force -Confirm:$false -ErrorAction Stop
                Write-Host "  stopped $($proc.ProcessName) (pid $($proc.Id)) on port $port"
                $stopped++
            } catch {
                Write-Host "  could not stop pid $($proc.Id) on port ${port}: $_"
            }
        } else {
            Write-Host "  port $port is held by '$($proc.ProcessName)' (pid $($proc.Id)) - not a python process, leaving it alone"
        }
    }
}

if ($stopped -eq 0) { Write-Host "Nothing to stop - no AMMBA services were running." }
else { Write-Host "Done ($stopped process(es) stopped). Data was in-memory and is now gone." }
