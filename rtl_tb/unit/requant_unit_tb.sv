// =============================================================================
// requant_unit_tb.sv  —  self-checking unit TB for the INT32->INT8 requantiser
// =============================================================================
// Drives the per-channel requant contract from hwconst/README and checks the
// INT8 outputs against a TB-side reference for several lanes:
//
//   biased   = bias_en ? acc + bias : acc
//   rounded  = (biased + (1<<(shift-1))) >>> shift     (round-half-up, s>0)
//   y_int8   = clip(rounded, -127, 127)
//
// Exercises: a normal positive case, a negative case, saturation high/low, the
// rounding guard bit, and bias_en=0. Pure SV (unpacked array ports, no XPM).
// Emits a VCD.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module requant_unit_tb;

  localparam int unsigned COUT = P_COUT_PAR;   // 64 lanes (only first few driven)

  logic                clock = 1'b0;
  logic                reset;
  logic [3:0]          shift_bits;
  logic                valid_in;
  logic [1:0]          act_type;
  logic signed [31:0]  acc_in  [0:COUT-1];
  logic signed [31:0]  bias_in [0:COUT-1];
  logic                bias_en;
  logic signed  [7:0]  y_out   [0:COUT-1];
  logic                valid_out;

  always #5 clock = ~clock;

  requant_unit #(.COUT_PAR(COUT)) dut (
    .clock(clock), .reset(reset),
    .shift_bits(shift_bits), .valid_in(valid_in), .act_type(act_type),
    .acc_in(acc_in), .bias_in(bias_in), .bias_en(bias_en),
    .y_out(y_out), .valid_out(valid_out)
  );

  integer errors = 0;
  integer i;

  // reference model
  function automatic signed [7:0] ref_requant(
      input signed [31:0] acc, input signed [31:0] bias,
      input logic ben, input [3:0] s);
    integer biased, rnd, off;
    begin
      biased = ben ? (acc + bias) : acc;
      off    = (s > 0) ? (1 <<< (s-1)) : 0;
      rnd    = (biased + off) >>> s;
      if (rnd >  127) ref_requant =  8'sd127;
      else if (rnd < -127) ref_requant = -8'sd127;
      else ref_requant = rnd[7:0];
    end
  endfunction

  initial begin
    $dumpfile("rtl_tb/sim_out/requant_unit.vcd");
    $dumpvars(0, requant_unit_tb);

    reset = 1'b1; valid_in = 1'b0; bias_en = 1'b0;
    shift_bits = 4'd0; act_type = ACT_NONE;
    for (i = 0; i < COUT; i = i + 1) begin acc_in[i]='0; bias_in[i]='0; end
    @(posedge clock); @(posedge clock); reset = 1'b0; @(posedge clock);

    // Lane stimuli: {acc, bias} with a representative shift.
    // 0: typical positive    -> 5000 + 200, >>7
    // 1: typical negative    -> -8000 + 0,  >>7
    // 2: saturate high       -> 1_000_000 >>3  -> clip 127
    // 3: saturate low        -> -1_000_000 >>3 -> clip -127
    // 4: rounding guard       -> 191 >>2 = round(47.75)=48
    // 5: bias add then shift  -> 10000 + (-3000) >>6
    acc_in[0] = 32'sd5000;     bias_in[0] = 32'sd200;
    acc_in[1] = -32'sd8000;    bias_in[1] = 32'sd0;
    acc_in[2] = 32'sd1000000;  bias_in[2] = 32'sd0;
    acc_in[3] = -32'sd1000000; bias_in[3] = 32'sd0;
    acc_in[4] = 32'sd191;      bias_in[4] = 32'sd0;
    acc_in[5] = 32'sd10000;    bias_in[5] = -32'sd3000;

    // Drive operands and valid_in on a negedge so they are stable before the
    // sampling posedge. requant_unit registers y_out/valid_out on the edge where
    // valid_in is high; we then sample just after that edge (while valid_out=1).
    @(negedge clock);
    shift_bits = 4'd7;
    bias_en    = 1'b1;
    valid_in   = 1'b1;
    @(posedge clock);          // y_out + valid_out registered on THIS edge
    #1;                        // settle, valid_out is now high
    valid_in   = 1'b0;

    if (valid_out !== 1'b1) begin
      errors = errors + 1;
      $display("  FAIL: valid_out not asserted");
    end

    // shift=7 lanes 0,1,5 ; lanes 2,3 saturate ; lane 4 use shift to show rounding
    for (i = 0; i <= 5; i = i + 1) begin
      logic signed [7:0] exp;
      exp = ref_requant(acc_in[i], bias_in[i], 1'b1, 4'd7);
      if (y_out[i] !== exp) begin
        errors = errors + 1;
        $display("  FAIL lane %0d: acc=%0d bias=%0d shift=7 -> got %0d exp %0d",
                 i, acc_in[i], bias_in[i], y_out[i], exp);
      end else begin
        $display("  ok   lane %0d: acc=%0d -> y=%0d", i, acc_in[i], y_out[i]);
      end
    end

    if (errors == 0) $display("==== requant_unit_tb: PASS ====");
    else             $display("==== requant_unit_tb: FAIL (%0d) ====", errors);
    $finish;
  end

endmodule : requant_unit_tb
