# =============================================================================
# run_xsim.ps1 — compile + run the Silicon YOLO RTL testbenches under Vivado xsim
# =============================================================================
# Cognichip's DEPS.yml targets the Vivado xsim flow (xvlog / xelab / xsim). This
# script compiles the full RTL (per DEPS.yml order) plus the testbenches and runs
# them. It auto-discovers Vivado from common install paths or $env:XILINX_VIVADO.
#
# Usage (from repo root, in a PowerShell with Vivado on PATH or installed):
#   .\rtl_tb\run_xsim.ps1                 # top-level TB (yolov10n_accel_tb)
#   .\rtl_tb\run_xsim.ps1 -Unit           # all unit TBs
#   .\rtl_tb\run_xsim.ps1 -Tb csd_mac_slice_tb
#
# If Vivado is NOT installed, the script prints the exact commands it would run
# and exits — the open-source Icarus path (rtl_tb/run_icarus.* ) covers the
# datapath unit modules without Vivado. See rtl_tb/SIM_SHOWCASE.md.
# =============================================================================
param(
  [switch]$Unit,
  [string]$Tb = "yolov10n_accel_tb",
  [string]$GoldenId = "0000"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$SimOut = Join-Path $Root "rtl_tb\sim_out"
New-Item -ItemType Directory -Force -Path $SimOut | Out-Null

# ---- locate Vivado --------------------------------------------------------
function Find-Vivado {
  if ($env:XILINX_VIVADO -and (Test-Path "$env:XILINX_VIVADO\bin\xvlog.bat")) {
    return "$env:XILINX_VIVADO\bin"
  }
  $cmd = Get-Command xvlog -ErrorAction SilentlyContinue
  if ($cmd) { return (Split-Path -Parent $cmd.Source) }
  foreach ($base in @("C:\Xilinx\Vivado", "D:\Xilinx\Vivado", "C:\Program Files\Xilinx\Vivado")) {
    if (Test-Path $base) {
      $ver = Get-ChildItem $base -Directory | Sort-Object Name -Descending | Select-Object -First 1
      if ($ver -and (Test-Path "$($ver.FullName)\bin\xvlog.bat")) { return "$($ver.FullName)\bin" }
    }
  }
  return $null
}

# ---- RTL compile order (from DEPS.yml) ------------------------------------
$Rtl = @(
  "rtl\yolov10n_pkg.sv",
  "rtl\csd_mac_slice.sv",
  "rtl\sce_mac_array.sv",
  "rtl\weight_rom_bank.sv",
  "rtl\fm_pingpong_buf.sv",
  "rtl\silu_lut_rom.sv",
  "rtl\requant_unit.sv",
  "rtl\layer_scheduler.sv",
  "rtl\input_preprocessor.sv",
  "rtl\axi4l_csr.sv",
  "rtl\yolov10n_accel_top.sv"
)

# ---- testbench selection --------------------------------------------------
$UnitTbs = @(
  @{ top = "csd_mac_slice_tb"; srcs = @("rtl\csd_mac_slice.sv", "rtl_tb\unit\csd_mac_slice_tb.sv"); pkg = "rtl\yolov10n_pkg.sv" },
  @{ top = "requant_unit_tb";  srcs = @("rtl\requant_unit.sv",  "rtl_tb\unit\requant_unit_tb.sv");  pkg = "rtl\yolov10n_pkg.sv" },
  @{ top = "silu_lut_rom_tb";  srcs = @("rtl\silu_lut_rom.sv",  "rtl_tb\unit\silu_lut_rom_tb.sv");  pkg = $null },
  @{ top = "sce_mac_array_tb"; srcs = @("rtl\sce_mac_array.sv", "rtl_tb\unit\sce_mac_array_tb.sv"); pkg = "rtl\yolov10n_pkg.sv" }
)

$vivBin = Find-Vivado

# ---- export golden hex (needs the project venv python) --------------------
$Py = "D:\Projects\FPGA\genesys2\.venv\Scripts\python.exe"
if (Test-Path $Py) {
  Write-Host "== exporting golden hex (id=$GoldenId) =="
  & $Py "rtl_tb\export_golden_hex.py" "--id" $GoldenId
} else {
  Write-Warning "venv python not found; assuming rtl_tb\golden_hex\$GoldenId already exists"
}

function Invoke-Xsim($topName, $sources) {
  $glbl = ""  # add glbl.v if XPM/unisim primitives need it
  Write-Host "== xvlog =="
  & "$vivBin\xvlog.bat" -sv $sources
  if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }
  Write-Host "== xelab =="
  & "$vivBin\xelab.bat" -debug typical -L unisims_ver -L xpm $topName -s "${topName}_sim"
  if ($LASTEXITCODE -ne 0) { throw "xelab failed" }
  Write-Host "== xsim =="
  & "$vivBin\xsim.bat" "${topName}_sim" -runall
}

if (-not $vivBin) {
  Write-Warning "Vivado (xvlog/xelab/xsim) not found."
  Write-Host ""
  Write-Host "Would run, for the TOP-LEVEL TB:" -ForegroundColor Cyan
  Write-Host "  xvlog -sv $($Rtl -join ' ') rtl_tb\yolov10n_accel_tb.sv"
  Write-Host "  xelab -debug typical -L unisims_ver -L xpm yolov10n_accel_tb -s accel_sim"
  Write-Host "  xsim accel_sim -runall"
  Write-Host ""
  Write-Host "Open-source alternative (no Vivado needed) for the datapath unit modules:" -ForegroundColor Green
  Write-Host "  .\rtl_tb\run_icarus.ps1"
  exit 2
}

Push-Location $SimOut
try {
  if ($Unit) {
    foreach ($u in $UnitTbs) {
      Write-Host "`n######## $($u.top) ########" -ForegroundColor Cyan
      $srcs = @()
      if ($u.pkg) { $srcs += (Join-Path $Root $u.pkg) }
      foreach ($s in $u.srcs) { $srcs += (Join-Path $Root $s) }
      Invoke-Xsim $u.top $srcs
    }
  } else {
    Write-Host "`n######## $Tb (top-level) ########" -ForegroundColor Cyan
    $srcs = @()
    foreach ($s in $Rtl) { $srcs += (Join-Path $Root $s) }
    $srcs += (Join-Path $Root "rtl_tb\yolov10n_accel_tb.sv")
    Invoke-Xsim $Tb $srcs
  }
} finally {
  Pop-Location
}
