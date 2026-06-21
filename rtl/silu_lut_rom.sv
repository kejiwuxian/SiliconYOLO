// =============================================================================
// silu_lut_rom.sv  —  SiLU Activation via 256-Entry INT8 LUT ROM
// =============================================================================
// SiLU(x) = x * sigmoid(x)
//
// For INT8 activations the input range is [-128, 127].  The LUT is pre-
// computed offline (scripts/gen_silu_lut.py) and stored in
// hwconst/mem/silu_lut.mem (256 entries, 8-bit signed hex values).
//
// Mapping: addr = unsigned(x_in) = x_in[7:0] reinterpreted as 8-bit uint.
//   addr 0   -> x = 0
//   addr 127 -> x = 127
//   addr 128 -> x = -128
//   ...
//   addr 255 -> x = -1
//
// Read latency: 1 clock cycle.
// Width:        One instance covers one output channel.
// The caller (requant_unit / layer_scheduler) replicates P_COUT_PAR=64
// instances in parallel so all output channels complete in one cycle.
// =============================================================================

`timescale 1ns/1ps

module silu_lut_rom #(
  parameter string LUT_MEM_FILE = "hwconst/mem/silu_lut.mem"
)(
  input  logic       clock,
  input  logic       en,            // 1 = pass-through; 0 = bypass (identity)

  input  logic [7:0] x_in,          // INT8 activation (unsigned addr, signed data)
  output logic [7:0] y_out          // INT8 SiLU result
);

  // ---------------------------------------------------------------------------
  // 256 × 8-bit LUT, implemented as distributed or block RAM
  // (256 entries fit in two BRAM18 or ~256 LUT6 cells)
  // ---------------------------------------------------------------------------
  (* rom_style = "block" *)
  logic [7:0] lut [0:255];

  initial begin
    $readmemh(LUT_MEM_FILE, lut);
  end

  logic [7:0] lut_out;
  always_ff @(posedge clock) begin
    lut_out <= lut[x_in];
  end

  // Bypass: if activation is ACT_NONE the caller drives en=0, and y_out
  // returns x_in delayed by one cycle (identity path).
  logic [7:0] x_in_r;
  always_ff @(posedge clock) begin
    x_in_r <= x_in;
  end

  assign y_out = en ? lut_out : x_in_r;

endmodule : silu_lut_rom
