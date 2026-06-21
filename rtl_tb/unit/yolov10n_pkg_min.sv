// =============================================================================
// yolov10n_pkg_min.sv  —  minimal package for OPEN-SOURCE unit simulation
// =============================================================================
// Icarus Verilog 14 cannot elaborate the large packed-struct assignment-pattern
// CFG_ROM initializer in the production rtl/yolov10n_pkg.sv ("sorry: I do not
// know how to elaborate assignment patterns using old method"). That CFG_ROM is
// only needed by the layer_scheduler / top-level (which also need Xilinx XPM and
// therefore target Vivado xsim anyway).
//
// The datapath UNIT modules (csd_mac_slice, sce_mac_array, requant_unit) only
// use the scalar localparams + the act_t / op_t enums — NOT CFG_ROM. This
// drop-in package exposes exactly those so the unit TBs compile and run in
// Icarus. The values are identical to the production package.
//
// Used ONLY for open-source unit sims (rtl_tb/unit/*). The full-chip TB uses the
// real rtl/yolov10n_pkg.sv under Vivado xsim.
// =============================================================================

package yolov10n_pkg;

  // System parameters (identical to rtl/yolov10n_pkg.sv)
  localparam int unsigned CLK_FREQ_MHZ      = 200;
  localparam int unsigned P_MAC_WIDTH       = 1024;
  localparam int unsigned P_COUT_PAR        = 64;
  localparam int unsigned P_CIN_PAR         = 16;
  localparam int unsigned P_TILE_H          = 8;
  localparam int unsigned P_TILE_W          = 8;
  localparam int unsigned P_QUANT_SHIFT_MAX = 15;
  localparam int unsigned P_DETECTION_MAX   = 300;
  localparam int unsigned WEIGHT_ROM_DEPTH  = 2290288;
  localparam int unsigned BIAS_ROM_DEPTH    =    8976;
  localparam int unsigned N_CONV_LAYERS     =      83;
  localparam int unsigned N_NODES           =      88;

  // Enumerated types (identical)
  typedef enum logic [1:0] {
    ACT_NONE = 2'd0,
    ACT_SILU = 2'd1,
    ACT_RELU = 2'd2
  } act_t;

  typedef enum logic [1:0] {
    OP_CONV     = 2'd0,
    OP_MAXPOOL  = 2'd1,
    OP_UPSAMPLE = 2'd2
  } op_t;

  // NOTE: layer_cfg_t + CFG_ROM intentionally omitted — see header. The unit
  // datapath modules do not reference them.

endpackage : yolov10n_pkg
