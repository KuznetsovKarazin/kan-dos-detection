#Requires -Version 5.1
<#
.SYNOPSIS
    Build, flash, and collect benchmark results for LUT-KAN on ESP32-C3 SuperMini.
.PARAMETER Port    COM port, e.g. -Port COM4  (auto-detected if omitted)
.PARAMETER Collect Number of loop cycles (each = 1 ATTACK + 1 NORMAL).
.PARAMETER SkipBuild  Skip pio compilation.
.PARAMETER Monitor    Open serial monitor.
.EXAMPLE
    .\flash_lutkan.ps1 -Collect 20
    .\flash_lutkan.ps1 -Port COM4 -Collect 20
    .\flash_lutkan.ps1 -Monitor
    .\flash_lutkan.ps1 -SkipBuild -Collect 20
#>
param(
    [string]$Port       = "",
    [int]   $Collect    = 0,
    [switch]$SkipBuild,
    [switch]$Monitor
)

Set-StrictMode -Version 1
$ErrorActionPreference = "Stop"

function Write-Step { param($m) Write-Host "  >> $m" -ForegroundColor Cyan  }
function Write-OK   { param($m) Write-Host "  OK $m" -ForegroundColor Green }
function Write-Fail { param($m) Write-Host " ERR $m" -ForegroundColor Red   }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName   = "esp32_lut_kan"
$BaudRate  = 115200

# 1. Auto-detect port
if ($Port -eq "") {
    Write-Step "Auto-detecting ESP32-C3 COM port..."
    $candidates = @(Get-PnpDevice -Class Ports -Status OK -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -match "USB|CH34|CP21|FTDI|ESP32" } |
        ForEach-Object { if ($_.FriendlyName -match "(COM\d+)") { $Matches[1] } })
    if ($candidates.Count -eq 0) {
        Write-Fail "No serial device found. Use -Port COM<N>"
        Add-Type -AssemblyName System.IO.Ports
        Write-Host "Available ports: $([System.IO.Ports.SerialPort]::GetPortNames() -join ', ')"
        exit 1
    }
    $Port = $candidates[0]
    Write-OK "Using port: $Port"
}

# 2. Build + Upload
if (-not $SkipBuild) {
    Write-Step "Building LUT-KAN firmware..."
    Push-Location $ScriptDir
    & pio run -e $EnvName -t upload --upload-port $Port
    if ($LASTEXITCODE -ne 0) { Write-Fail "Build/upload failed"; exit 1 }
    Pop-Location
    Write-OK "Flashed successfully"
} else {
    Write-Step "Skipping build (-SkipBuild)"
}

Start-Sleep -Seconds 2

# 3. Serial monitor or collection
if (($Collect -gt 0) -or $Monitor) {
    Write-Step "Opening serial port $Port at $BaudRate baud..."
    Add-Type -AssemblyName System.IO.Ports
    $sp = [System.IO.Ports.SerialPort]::new($Port, $BaudRate)
    $sp.ReadTimeout = 8000
    $sp.Open()
    Start-Sleep -Milliseconds 800

    if ($Monitor) {
        Write-OK "Monitor mode — press Ctrl+C to exit"
        try { while ($true) { Write-Host $sp.ReadLine().Trim() } } catch { }
        $sp.Close(); exit 0
    }

    $rows_needed = $Collect * 2
    Write-Step "Collecting $rows_needed rows..."

    $rows = [System.Collections.ArrayList]@()
    $meta = [System.Collections.ArrayList]@()
    $deadline = (Get-Date).AddSeconds(180)

    while (($rows.Count -lt $rows_needed) -and ((Get-Date) -lt $deadline)) {
        try {
            $line = $sp.ReadLine().Trim()
            if ($line -match "^(ATTACK|NORMAL),") {
                [void]$rows.Add($line)
                $pct = [int]($rows.Count * 100 / $rows_needed)
                Write-Progress -Activity "Collecting" -Status "$($rows.Count) / $rows_needed" -PercentComplete $pct
            } elseif ($line -ne "") {
                [void]$meta.Add($line)
            }
        } catch [System.TimeoutException] { continue }
    }
    Write-Progress -Activity "Collecting" -Completed
    $sp.Close()

    if ($rows.Count -eq 0) {
        Write-Fail "No data rows. Use -Monitor to debug first."
        exit 1
    }

    # 4. Statistics
    $latencies = [System.Collections.Generic.List[long]]@()
    $correct = 0; $total = 0
    foreach ($r in $rows) {
        $p = $r -split ","
        if ($p.Count -ge 6) {
            $latencies.Add([long]$p[3])
            $correct += [int]$p[5]
            $total++
        }
    }

    $mean_us = ($latencies | Measure-Object -Average).Average
    $min_us  = ($latencies | Measure-Object -Minimum).Minimum
    $max_us  = ($latencies | Measure-Object -Maximum).Maximum
    $variance = ($latencies | ForEach-Object { [math]::Pow($_ - $mean_us, 2) } | Measure-Object -Average).Average
    $std_us  = [math]::Sqrt($variance)
    $hw_acc  = if ($total -gt 0) { [math]::Round($correct / $total, 4) } else { 0.0 }

    $model_name = ($meta | Where-Object { $_ -match "^MODEL=" } | Select-Object -Last 1) -replace "MODEL=", ""
    $lut_l      = ($meta | Where-Object { $_ -match "^LUT_L=" } | Select-Object -Last 1) -replace "LUT_L=", ""
    $f1_val     = ($meta | Where-Object { $_ -match "^F1=" }    | Select-Object -Last 1) -replace "F1=", ""
    $roc_val    = ($meta | Where-Object { $_ -match "^ROC_AUC=" }| Select-Object -Last 1) -replace "ROC_AUC=", ""
    if ($model_name -eq "") { $model_name = "LUT_KAN" }

    $mean_s = "{0:F1}" -f $mean_us
    $std_s  = "{0:F1}" -f $std_us
    $ts     = Get-Date -Format "yyyyMMdd_HHmmss"
    $board  = "ESP32-C3-SuperMini"

    # 5. Print results
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  HARDWARE BENCHMARK RESULTS"
    Write-Host "============================================================"
    Write-Host "  Timestamp  : $ts"
    Write-Host "  Model      : $model_name (L=$lut_l)"
    Write-Host "  Board      : $board"
    Write-Host "  F1 (test)  : $f1_val     ROC-AUC : $roc_val"
    Write-Host "  Samples    : $total   Accuracy on hw : $hw_acc"
    Write-Host ""
    Write-Host "  Latency per inference:"
    Write-Host "    mean = $mean_s us    std = $std_s us"
    Write-Host "    min  = $min_us us    max = $max_us us"
    Write-Host ""
    Write-Host "  PAPER-READY BLOCK"
    Write-Host "  Model: $model_name (L=$lut_l) | Board: $board"
    Write-Host "  Latency: $mean_s +/- $std_s us (min $min_us, max $max_us)"
    Write-Host "  SRAM runtime: under 512 B (model in Flash)"
    Write-Host "  F1: $f1_val  ROC-AUC: $roc_val"
    Write-Host "============================================================"
    Write-Host ""

    # 6. Save report
    $results_dir = Join-Path $ScriptDir "results"
    New-Item -ItemType Directory -Path $results_dir -Force | Out-Null
    $fname = "${ts}_${board}_${model_name}_L${lut_l}_report.txt"
    $report_path = Join-Path $results_dir $fname

    $lines = @(
        "============================================================",
        "  HARDWARE BENCHMARK RESULTS",
        "============================================================",
        "  Timestamp  : $ts",
        "  Model      : $model_name (L=$lut_l)",
        "  Board      : $board",
        "  F1 (test)  : $f1_val     ROC-AUC : $roc_val",
        "  Samples    : $total   Accuracy on hw : $hw_acc",
        "",
        "  Latency per inference:",
        "    mean = $mean_s us    std = $std_s us",
        "    min  = $min_us us    max = $max_us us",
        "",
        "  PAPER-READY BLOCK",
        "  Model: $model_name (L=$lut_l) | Board: $board",
        "  Latency: $mean_s +/- $std_s us (min $min_us, max $max_us)",
        "  SRAM runtime: under 512 B (model in Flash)",
        "  F1: $f1_val  ROC-AUC: $roc_val",
        "============================================================"
    )
    $lines | Out-File $report_path -Encoding UTF8
    Write-OK "Report saved: $report_path"
}
