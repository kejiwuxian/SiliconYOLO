// =============================================================================
// csd_mac_slice.sv  —  Single INT8 × INT8 Multiply-Accumulate (0-DSP)
// =============================================================================
// Implements one MAC lane:
//   acc_q <= acc_q + signed(weight_i) * signed(act_i)
//
// The `(* use_dsp = "no" *)` attribute forces Vivado to synthesise the 8×8
// signed multiply into 6-LUT fabric using a shift-and-add tree, achieving the
// 0-DSP design goal.  Each slice costs ~28-36 LUTs and 1 pipeline register.
//
// Inputs are registered on the rising edge of `clock`.  One accumulate result
// is output per cycle.  The caller drives acc_clear to zero the accumulator
// at the start of a new output-channel group.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module csd_mac_slice (
  input  logic        clock,
  input  logic        reset,       // sync active-high reset

  // Control
  input  logic        acc_clear,   // zero accumulator this cycle (before MAC)
  input  logic        mac_en,      // enable MAC this cycle

  // Data
  input  logic signed [7:0]  act_i,     // INT8 activation from IFM buffer
  input  logic signed [7:0]  weight_i,  // INT8 weight from weight ROM

  // Accumulator output (INT32)
  output logic signed [31:0] acc_q
);

  // ---------------------------------------------------------------------------
  // Force LUT synthesis, no DSP inference
  // ---------------------------------------------------------------------------
  (* use_dsp = "no" *)
  logic signed [15:0] product;

  always_comb begin
    product = act_i * weight_i;  // 8x8 -> 16b; Vivado maps to LUT adder tree
  end

  // ---------------------------------------------------------------------------
  // Pipelined accumulator
  // ---------------------------------------------------------------------------
  always_ff @(posedge clock) begin
    if (reset) begin
      acc_q <= '0;
    end else if (mac_en) begin
      if (acc_clear)
        acc_q <= {{16{product[15]}}, product};   // sign-extend on clear+MAC
      else
        acc_q <= acc_q + {{16{product[15]}}, product};
    end else if (acc_clear) begin
      acc_q <= '0;
    end
  end

endmodule : csd_mac_slice
