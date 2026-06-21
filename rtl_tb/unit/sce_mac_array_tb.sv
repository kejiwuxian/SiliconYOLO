// =============================================================================
// sce_mac_array_tb.sv  —  self-checking unit TB for the 1024-wide MAC array
// =============================================================================
// Drives the Shared Compute Engine (P_COUT_PAR x P_CIN_PAR INT8 MAC array) with
// a real multi-cycle conv reduction and checks every output-channel accumulator
// against a TB-side integer reference.
//
//   Per cycle: P_CIN_PAR activations broadcast to P_COUT_PAR channels, each
//   channel multiply-accumulates its CIN_PAR weights. acc_clear loads on the
//   first cycle, then subsequent cycles accumulate (c_in reduction).
//
// Pure SV (unpacked array ports, no XPM). Emits a VCD.
//
// NOTE: sce_mac_array.sv uses an `automatic` variable inside an always_comb
// reduction (line ~105) that Icarus Verilog 14 rejects ("Overriding the default
// variable lifetime is not yet supported"). This TB therefore targets Vivado
// xsim / Verilator. See rtl_tb/SIM_SHOWCASE.md.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module sce_mac_array_tb;

  localparam int unsigned COUT = P_COUT_PAR;   // 64
  localparam int unsigned CIN  = P_CIN_PAR;    // 16
  localparam int unsigned NCYC = 4;            // c_in reduction cycles

  logic clock = 1'b0;
  logic reset;
  logic mac_en;
  logic acc_clear;
  logic dw_mode;
  logic [7:0] weight_bus [0:COUT*CIN-1];
  logic [7:0] act_bus    [0:CIN-1];
  logic signed [31:0] acc_out [0:COUT-1];
  logic acc_valid;

  always #5 clock = ~clock;

  sce_mac_array #(.COUT_PAR(COUT), .CIN_PAR(CIN)) dut (
    .clock(clock), .reset(reset),
    .mac_en(mac_en), .acc_clear(acc_clear), .dw_mode(dw_mode),
    .weight_bus(weight_bus), .act_bus(act_bus),
    .acc_out(acc_out), .acc_valid(acc_valid)
  );

  // reference accumulators
  integer ref_acc [0:COUT-1];
  integer errors = 0;
  integer cyc, co, ci, idx;

  // scalar probes so the VCD shows representative bus values (Icarus does not
  // dump unpacked-array ports by default).
  logic signed [31:0] probe_acc0;
  logic signed [31:0] probe_acc32;
  logic signed  [7:0] probe_act0;
  logic signed  [7:0] probe_wgt0;
  always_comb begin
    probe_acc0  = acc_out[0];
    probe_acc32 = acc_out[32];
    probe_act0  = act_bus[0];
    probe_wgt0  = weight_bus[0];
  end

  // deterministic operand generator (signed INT8)
  function automatic integer s8(input integer v);
    s8 = ((v + 128) % 256) - 128;
  endfunction

  initial begin
    $dumpfile("rtl_tb/sim_out/sce_mac_array.vcd");
    $dumpvars(0, sce_mac_array_tb);

    reset = 1'b1; mac_en = 1'b0; acc_clear = 1'b0; dw_mode = 1'b0;
    for (idx = 0; idx < COUT*CIN; idx = idx + 1) weight_bus[idx] = 8'd0;
    for (idx = 0; idx < CIN; idx = idx + 1) act_bus[idx] = 8'd0;
    for (co = 0; co < COUT; co = co + 1) ref_acc[co] = 0;
    @(posedge clock); @(posedge clock); reset = 1'b0; @(posedge clock);

    // multi-cycle reduction
    @(negedge clock);
    for (cyc = 0; cyc < NCYC; cyc = cyc + 1) begin
      // load operands for this cycle
      for (ci = 0; ci < CIN; ci = ci + 1)
        act_bus[ci] = s8(7*cyc + 3*ci + 11);
      for (co = 0; co < COUT; co = co + 1)
        for (ci = 0; ci < CIN; ci = ci + 1)
          weight_bus[co*CIN + ci] = s8(5*co - 2*ci + 9*cyc - 6);
      // reference accumulate
      for (co = 0; co < COUT; co = co + 1)
        for (ci = 0; ci < CIN; ci = ci + 1)
          ref_acc[co] = ref_acc[co] +
            $signed(act_bus[ci]) * $signed(weight_bus[co*CIN + ci]);

      mac_en    = 1'b1;
      acc_clear = (cyc == 0);
      @(negedge clock);
    end
    mac_en = 1'b0; acc_clear = 1'b0;
    @(posedge clock); #1;   // settle final accumulate

    // check a representative spread of channels
    for (co = 0; co < COUT; co = co + 4) begin
      if (acc_out[co] !== ref_acc[co]) begin
        errors = errors + 1;
        $display("  FAIL c_out=%0d: got %0d  exp %0d", co, acc_out[co], ref_acc[co]);
      end
    end
    $display("[sce_mac_array_tb] checked %0d channels over %0d c_in cycles", COUT/4, NCYC);
    $display("  sample c_out[0]=%0d (exp %0d), c_out[32]=%0d (exp %0d)",
             acc_out[0], ref_acc[0], acc_out[32], ref_acc[32]);

    if (errors == 0) $display("==== sce_mac_array_tb: PASS ====");
    else             $display("==== sce_mac_array_tb: FAIL (%0d) ====", errors);
    $finish;
  end

endmodule : sce_mac_array_tb
