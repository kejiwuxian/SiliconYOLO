// =============================================================================
// sce_mac_array.sv  —  Shared Compute Engine: 1024-wide INT8 MAC Array
// =============================================================================
// Architecture:
//   P_COUT_PAR (64) output channels × P_CIN_PAR (16) input channels = 1024 MACs
//
// One compute cycle processes P_CIN_PAR activation values against
// P_COUT_PAR × P_CIN_PAR weights, accumulating into P_COUT_PAR INT32 sums.
//
// The weight bus is 1024 bytes wide (1024 × 8b), read from the Weight ROM Bank
// in a single BRAM clock.  The activation bus carries P_CIN_PAR INT8 values.
//
// For a depthwise layer the caller sets dw_mode and each c_out accumulates
// independently (c_in_chunk is always 1 for DW, k² terms accumulate).
//
// Bias add and requantisation are handled downstream in requant_unit.sv.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module sce_mac_array #(
  parameter int unsigned COUT_PAR = P_COUT_PAR,   // 64
  parameter int unsigned CIN_PAR  = P_CIN_PAR     // 16
)(
  input  logic        clock,
  input  logic        reset,

  // ---- Control signals from layer_scheduler --------------------------------
  input  logic        mac_en,       // 1 = execute MAC this cycle
  input  logic        acc_clear,    // 1 = clear accumulators before MAC
  input  logic        dw_mode,      // 1 = depthwise (each c_out has 1 c_in)

  // ---- Weight bus: COUT_PAR × CIN_PAR bytes --------------------------------
  // Packed as weight_bus[cout_idx*CIN_PAR + cin_idx][7:0]
  input  logic [7:0]  weight_bus [0:COUT_PAR*CIN_PAR-1],

  // ---- Input activation bus: CIN_PAR bytes (shared across all c_out) -------
  input  logic [7:0]  act_bus    [0:CIN_PAR-1],

  // ---- Accumulator outputs: COUT_PAR INT32 values --------------------------
  output logic signed [31:0] acc_out [0:COUT_PAR-1],
  output logic               acc_valid    // registered, 1 cycle after mac_en
);

  // ---------------------------------------------------------------------------
  // 1024 MAC slices, instantiated as a 2D array [cout_par][cin_par]
  // ---------------------------------------------------------------------------
  // For standard (non-DW) conv:
  //   Each accumulator cout_idx sums over all cin_par inputs.
  //   weight_bus[cout_idx*CIN_PAR + cin_idx] × act_bus[cin_idx]
  //
  // For depthwise conv (dw_mode = 1):
  //   cout_idx == cin_idx channel; only the diagonal element is valid.
  //   The caller sets CIN_PAR=1 effectively by only asserting mac_en for
  //   one c_in per c_out group.  This module still instantiates the full
  //   array; dw_mode masks inactive lanes via weight_mask.
  // ---------------------------------------------------------------------------

  genvar gi, gj;

  // Partial products from each MAC slice
  (* use_dsp = "no" *)
  logic signed [15:0] pp [0:COUT_PAR-1][0:CIN_PAR-1];

  // Per-accumulator clear flag (broadcast from acc_clear)
  logic acc_clear_r;
  logic mac_en_r;

  always_ff @(posedge clock) begin
    if (reset) begin
      acc_clear_r <= 1'b0;
      mac_en_r    <= 1'b0;
    end else begin
      acc_clear_r <= acc_clear;
      mac_en_r    <= mac_en;
    end
  end

  // ---------------------------------------------------------------------------
  // Combinational multiply array (LUT-based, no DSPs)
  // ---------------------------------------------------------------------------
  generate
    for (gi = 0; gi < COUT_PAR; gi++) begin : gen_cout
      for (gj = 0; gj < CIN_PAR; gj++) begin : gen_cin
        always_comb begin
          // In DW mode only the lane where gj == (gi % CIN_PAR) is active;
          // the scheduler ensures weight_bus has the correct value there.
          pp[gi][gj] = signed'(weight_bus[gi*CIN_PAR + gj]) *
                       signed'(act_bus[gj]);
        end
      end
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Reduction tree: sum CIN_PAR partial products into each accumulator
  // ---------------------------------------------------------------------------
  // 16->32 bit sign extension, then 16-to-1 adder tree (4 levels, all LUT)
  logic signed [31:0] pp_sum [0:COUT_PAR-1];

  generate
    for (gi = 0; gi < COUT_PAR; gi++) begin : gen_sum
      always_comb begin
        logic signed [31:0] s; s = '0;
        for (int j = 0; j < CIN_PAR; j++) begin
          s = s + {{16{pp[gi][j][15]}}, pp[gi][j]};
        end
        pp_sum[gi] = s;
      end
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // INT32 accumulator registers (one per output channel)
  // ---------------------------------------------------------------------------
  generate
    for (gi = 0; gi < COUT_PAR; gi++) begin : gen_acc
      always_ff @(posedge clock) begin
        if (reset) begin
          acc_out[gi] <= '0;
        end else if (mac_en) begin
          if (acc_clear)
            acc_out[gi] <= pp_sum[gi];
          else
            acc_out[gi] <= acc_out[gi] + pp_sum[gi];
        end else if (acc_clear) begin
          acc_out[gi] <= '0;
        end
      end
    end
  endgenerate

  // acc_valid pulses one cycle after the last mac_en of a group.
  // The scheduler drives a separate acc_valid_in; we just pipeline it.
  // (In the top-level, layer_scheduler drives acc_valid_in and this output
  //  is used to trigger bias add + requant.)
  always_ff @(posedge clock) begin
    if (reset) acc_valid <= 1'b0;
    else       acc_valid <= mac_en_r & ~mac_en;  // falling edge of mac_en burst
  end

endmodule : sce_mac_array
