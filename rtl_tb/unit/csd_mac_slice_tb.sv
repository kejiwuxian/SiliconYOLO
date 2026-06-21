// =============================================================================
// csd_mac_slice_tb.sv  —  self-checking unit TB for the INT8 MAC lane
// =============================================================================
// Drives a real INT8 dot product through csd_mac_slice and checks the INT32
// accumulator against a reference computed in the testbench. Emits a VCD so the
// waveform can be rendered ("the chip computing a dot product").
//
// Simulator-agnostic: pure SystemVerilog, no XPM, no struct ports.
//   iverilog -g2012 -s csd_mac_slice_tb -o sim.vvp \
//     rtl/yolov10n_pkg.sv rtl/csd_mac_slice.sv rtl_tb/unit/csd_mac_slice_tb.sv
//   vvp sim.vvp
// =============================================================================

`timescale 1ns/1ps

module csd_mac_slice_tb;

  // ---- DUT I/O --------------------------------------------------------------
  logic              clock = 1'b0;
  logic              reset;
  logic              acc_clear;
  logic              mac_en;
  logic signed [7:0] act_i;
  logic signed [7:0] weight_i;
  logic signed [31:0] acc_q;

  // 10 ns clock
  always #5 clock = ~clock;

  // ---- DUT ------------------------------------------------------------------
  csd_mac_slice dut (
    .clock    (clock),
    .reset    (reset),
    .acc_clear(acc_clear),
    .mac_en   (mac_en),
    .act_i    (act_i),
    .weight_i (weight_i),
    .acc_q    (acc_q)
  );

  // ---- Stimulus: a real 8-tap INT8 dot product ------------------------------
  localparam int N = 8;
  // activations / weights resembling a quantized conv reduction
  logic signed [7:0] acts [0:N-1];
  logic signed [7:0] wgts [0:N-1];

  integer i;
  integer ref_acc;
  integer errors = 0;

  initial begin
    $dumpfile("rtl_tb/sim_out/csd_mac_slice.vcd");
    $dumpvars(0, csd_mac_slice_tb);

    // deterministic test vector (mix of signs / magnitudes)
    acts = '{  8'sd72, -8'sd40,  8'sd15,  8'sd120, -8'sd96,  8'sd33, -8'sd8,  8'sd64 };
    wgts = '{ -8'sd12,  8'sd55,  8'sd80, -8'sd24,  8'sd17, -8'sd100, 8'sd64, 8'sd31 };

    // reference dot product
    ref_acc = 0;
    for (i = 0; i < N; i = i + 1) ref_acc = ref_acc + (acts[i] * wgts[i]);

    // reset
    reset = 1'b1; acc_clear = 1'b0; mac_en = 1'b0;
    act_i = 8'sd0; weight_i = 8'sd0;
    @(posedge clock); @(posedge clock);
    reset = 1'b0;
    @(posedge clock);

    // Feed the taps. csd_mac_slice samples `act_i`/`weight_i` (combinational
    // `product`) on the rising edge, so we drive each operand pair and pulse one
    // clock per tap. The first tap asserts acc_clear to LOAD (clear+accumulate)
    // rather than add to stale data. We change inputs just after each edge
    // (blocking assigns at the negedge region via #1) so they are stable well
    // before the next sampling edge.
    @(negedge clock);
    for (i = 0; i < N; i = i + 1) begin
      act_i     = acts[i];
      weight_i  = wgts[i];
      mac_en    = 1'b1;
      acc_clear = (i == 0);
      @(negedge clock);   // operands set on this negedge, sampled on next posedge
    end
    // hold one more enabled cycle so the final tap's product is accumulated,
    // then drop the enable.
    mac_en    = 1'b0;
    acc_clear = 1'b0;
    @(posedge clock);   // settle the final accumulate into acc_q
    #1;

    // ---- check ----
    $display("[csd_mac_slice_tb] DUT acc_q = %0d   reference = %0d", acc_q, ref_acc);
    if (acc_q !== ref_acc) begin
      errors = errors + 1;
      $display("  FAIL: mismatch (got %0d, exp %0d)", acc_q, ref_acc);
    end else begin
      $display("  PASS: INT8 8-tap dot product matches integer reference");
    end

    if (errors == 0) $display("==== csd_mac_slice_tb: PASS ====");
    else             $display("==== csd_mac_slice_tb: FAIL (%0d) ====", errors);
    $finish;
  end

endmodule : csd_mac_slice_tb
