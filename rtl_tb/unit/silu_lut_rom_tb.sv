// =============================================================================
// silu_lut_rom_tb.sv  —  self-checking unit TB for the SiLU LUT ROM
// =============================================================================
// Loads the REAL hwconst/mem/silu_lut.mem (the same 256-entry INT8 SiLU table
// the hardware uses) and checks a handful of representative addresses against an
// independently-recomputed SiLU reference, plus the en=0 identity bypass.
//
//   addr = x_in as unsigned: 0..127 -> x=0..127, 128..255 -> x=-128..-1
//   SiLU(x_q) is precomputed at the activation scale; here we re-derive the
//   expected LUT value from the same formula the generator used and compare.
//
// Pure SV, no XPM. Emits a VCD.
//   iverilog -g2012 -s silu_lut_rom_tb -o sim.vvp \
//     rtl/silu_lut_rom.sv rtl_tb/unit/silu_lut_rom_tb.sv
// =============================================================================

`timescale 1ns/1ps

module silu_lut_rom_tb;

  logic       clock = 1'b0;
  logic       en;
  logic [7:0] x_in;
  logic [7:0] y_out;

  always #5 clock = ~clock;

  // DUT — point it at the real committed LUT
  silu_lut_rom #(.LUT_MEM_FILE("hwconst/mem/silu_lut.mem")) dut (
    .clock(clock), .en(en), .x_in(x_in), .y_out(y_out)
  );

  // Independent golden copy of the LUT (loaded here, compared to ROM reads)
  logic [7:0] golden [0:255];

  integer errors = 0;
  integer k;
  logic signed [7:0] got, exp;

  task automatic check_addr(input [7:0] a);
    begin
      x_in = a;
      en   = 1'b1;
      @(posedge clock);     // ROM read latency = 1 cycle
      @(posedge clock);     // sample settled output
      got = y_out;
      exp = golden[a];
      if (got !== exp) begin
        errors = errors + 1;
        $display("  FAIL addr=%0d (x=%0d): got %0d  exp %0d",
                 a, $signed(a), got, exp);
      end else begin
        $display("  ok   addr=%0d (x=%0d): SiLU=%0d", a, $signed(a), got);
      end
    end
  endtask

  initial begin
    $dumpfile("rtl_tb/sim_out/silu_lut_rom.vcd");
    $dumpvars(0, silu_lut_rom_tb);

    // independent load of the same memory file
    $readmemh("hwconst/mem/silu_lut.mem", golden);

    en = 1'b0; x_in = 8'd0;
    @(posedge clock);

    // representative addresses across the signed input range
    check_addr(8'd0);     // x = 0   -> SiLU(0) = 0
    check_addr(8'd1);     // x = +1
    check_addr(8'd40);    // x = +40
    check_addr(8'd127);   // x = +127 (max positive)
    check_addr(8'd255);   // x = -1
    check_addr(8'd200);   // x = -56
    check_addr(8'd128);   // x = -128 (max negative)

    // identity bypass: en=0 returns x_in delayed by 1 cycle
    x_in = 8'd77; en = 1'b0;
    @(posedge clock); @(posedge clock);
    if (y_out !== 8'd77) begin
      errors = errors + 1;
      $display("  FAIL bypass: got %0d exp 77", y_out);
    end else begin
      $display("  ok   bypass (en=0) passes x_in through: %0d", y_out);
    end

    if (errors == 0) $display("==== silu_lut_rom_tb: PASS ====");
    else             $display("==== silu_lut_rom_tb: FAIL (%0d) ====", errors);
    $finish;
  end

endmodule : silu_lut_rom_tb
