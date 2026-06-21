# =============================================================================
# run_icarus.ps1 — compile + run the datapath UNIT testbenches in Icarus Verilog
# =============================================================================
# Open-source simulation path (NO Vivado required). Runs the pure-SV datapath
# unit TBs that don't need Xilinx XPM, dumps a VCD per module into
# rtl_tb/sim_out/, and prints PASS/FAIL.
#
# Modules csd_mac_slice + silu_lut_rom run against the ORIGINAL rtl/ sources.
# requant_unit + sce_mac_array use an `automatic` variable inside always_comb
# that Icarus 14 rejects, so this script compiles them from rtl_tb/sim_shim/
# (identical logic, `automatic` lifetime keyword removed — see SIM_SHOWCASE.md).
#
# Requires Icarus (iverilog/vvp). Point $env:OSSCAD at an OSS CAD Suite install,
# or have iverilog on PATH.
#
#   .\rtl_tb\run_icarus.ps1
# =============================================================================
param(
  [string]$Osscad = $env:OSSCAD
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$SimOut = Join-Path $Root "rtl_tb\sim_out"
New-Item -ItemType Directory -Force -Path $SimOut | Out-Null

# ---- locate iverilog/vvp --------------------------------------------------
$iverilog = $null; $vvp = $null; $libDir = $null
if ($Osscad -and (Test-Path "$Osscad\bin\iverilog.exe")) {
  $iverilog = "$Osscad\bin\iverilog.exe"; $vvp = "$Osscad\bin\vvp.exe"; $libDir = "$Osscad\lib"
} else {
  $c = Get-Command iverilog -ErrorAction SilentlyContinue
  if ($c) { $iverilog = $c.Source; $vvp = "vvp" }
}
if (-not $iverilog) {
  Write-Error "iverilog not found. Install OSS CAD Suite and set `$env:OSSCAD, or put iverilog on PATH."
  exit 1
}
# OSS CAD Suite ships its DLLs in lib/; put it first so vvp resolves libreadline etc.
if ($libDir) { $env:PATH = "$libDir;$(Split-Path $iverilog);$env:PATH" }

$Pkg = Join-Path $Root "rtl_tb\unit\yolov10n_pkg_min.sv"

# top, sources (relative to repo root)
$Tbs = @(
  @{ top="csd_mac_slice_tb"; srcs=@("rtl\csd_mac_slice.sv","rtl_tb\unit\csd_mac_slice_tb.sv") },
  @{ top="silu_lut_rom_tb";  srcs=@("rtl\silu_lut_rom.sv","rtl_tb\unit\silu_lut_rom_tb.sv"); noPkg=$true },
  @{ top="requant_unit_tb";  srcs=@("rtl_tb\sim_shim\requant_unit.sv","rtl_tb\unit\requant_unit_tb.sv") },
  @{ top="sce_mac_array_tb"; srcs=@("rtl_tb\sim_shim\sce_mac_array.sv","rtl_tb\unit\sce_mac_array_tb.sv") }
)

$pass = 0; $fail = 0
Push-Location $Root
try {
  foreach ($t in $Tbs) {
    $vvpFile = Join-Path $SimOut "$($t.top).vvp"
    $logFile = Join-Path $SimOut "$($t.top).log"
    $srcs = @()
    if (-not $t.noPkg) { $srcs += $Pkg }
    $srcs += $t.srcs
    Write-Host "`n######## $($t.top) ########" -ForegroundColor Cyan
    & $iverilog -g2012 -s $t.top -o $vvpFile @srcs 2>&1 | Tee-Object -FilePath $logFile | Out-Null
    & $vvp $vvpFile 2>&1 | Tee-Object -FilePath $logFile -Append | ForEach-Object { Write-Host $_ }
    if ((Get-Content $logFile -Raw) -match "PASS ====") { $pass++ } else { $fail++ }
  }
} finally { Pop-Location }

Write-Host "`n==== Icarus unit suite: $pass passed, $fail failed ====" -ForegroundColor $(if ($fail -eq 0) {"Green"} else {"Red"})
Write-Host "VCDs in rtl_tb\sim_out\*.vcd  ->  render with: python tools\vcd_to_png.py ..."
exit $fail
