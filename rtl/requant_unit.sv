// =============================================================================
// requant_unit.sv  —  INT32 -> INT8 Requantisation (P_COUT_PAR lanes)
// =============================================================================
// Implements the per-output-channel requantisation contract from hwconst/README:
//
//   acc_int32 += bias_int32            (bias already INT32 in ROM)
//   y_int8 = clip(round(acc >> shift_bits), -127, 127)
//
// The shift_bits value comes from the layer config (CFG_ROM[node].shift_bits).
// All P_COUT_PAR (64) channels are requantised in parallel.
//
// Rounding: arithmetic right-shift with round-half-up:
//   round(x >> s) = (x + (1 << (s-1))) >> s   for s > 0
//                 =  x                          for s = 0
//
// Saturation: output is clipped to [-127, 127] (symmetric INT8, not -128..127,
// to match the PTQ quantisation scheme used during calibration).
//
// Pipeline latency: 1 clock cycle from valid acc + bias inputs to y_int8.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module requant_unit #(
  parameter int unsigned COUT_PAR = P_COUT_PAR   // 64 lanes
)(
  input  logic        clock,
  input  logic        reset,

  // ---- Per-layer control (from layer_scheduler) ----------------------------
  input  logic [3:0]  shift_bits,   // 0-15 right-shift
  input  logic        valid_in,     // 1 = acc + bias inputs are valid
  input  logic [1:0]  act_type,     // from ACT_* enum (passed to SiLU)

  // ---- Accumulator + bias inputs -------------------------------------------
  input  logic signed [31:0] acc_in   [0:COUT_PAR-1],
  input  logic signed [31:0] bias_in  [0:COUT_PAR-1],
  input  logic               bias_en, // 1 = add bias_in; 0 = no bias

  // ---- INT8 output ---------------------------------------------------------
  output logic signed  [7:0] y_out    [0:COUT_PAR-1],
  output logic               valid_out
);

  // ---------------------------------------------------------------------------
  // Stage 1: bias add
  // ---------------------------------------------------------------------------
  logic signed [31:0] biased [0:COUT_PAR-1];

  always_comb begin
    for (int i = 0; i < COUT_PAR; i++) begin
      biased[i] = bias_en ? (acc_in[i] + bias_in[i]) : acc_in[i];
    end
  end

  // ---------------------------------------------------------------------------
  // Stage 2: rounding arithmetic right-shift + saturate, registered
  // ---------------------------------------------------------------------------
  // Round-half-up: add 2^(shift-1) before shifting (guard bit)
  logic signed [31:0] rounded [0:COUT_PAR-1];
  logic signed  [7:0] clipped [0:COUT_PAR-1];

  always_comb begin
    for (int i = 0; i < COUT_PAR; i++) begin
      // Apply rounding offset (only when shift_bits > 0)
      automatic logic signed [31:0] rnd_offset;
      rnd_offset = (shift_bits > 4'd0) ? (32'b1 <<< (shift_bits - 4'd1)) : 32'b0;
      rounded[i] = (biased[i] + rnd_offset) >>> shift_bits;

      // Saturate to [-127, 127]
      if (rounded[i] > 32'sd127)
        clipped[i] = 8'sd127;
      else if (rounded[i] < -32'sd127)
        clipped[i] = -8'sd127;
      else
        clipped[i] = rounded[i][7:0];
    end
  end

  // Register output
  always_ff @(posedge clock) begin
    if (reset) begin
      valid_out <= 1'b0;
      for (int i = 0; i < COUT_PAR; i++)
        y_out[i] <= '0;
    end else if (valid_in) begin
      valid_out <= 1'b1;
      for (int i = 0; i < COUT_PAR; i++)
        y_out[i] <= clipped[i];
    end else begin
      valid_out <= 1'b0;
    end
  end

endmodule : requant_unit
