// =============================================================================
// fm_pingpong_buf.sv  —  Feature-Map Ping-Pong Buffer
// =============================================================================
// Dual-bank on-chip SRAM buffer for feature maps.
// While Bank A is being read by the SCE (consumer), Bank B is being written
// by the output of the previous layer (producer), and vice versa.
//
// Capacity per bank:  DEPTH × 8 bits, implemented as RAMB36E1 block RAM.
// The caller (layer_scheduler) controls which bank is "write active" via
// bank_sel; the read bank is automatically the other one.
//
// Address is a flat byte offset: [channel][row][col] linearised by the
// address generator in layer_scheduler.sv.
//
// Read latency:  1 clock (registered BRAM output)
// Write latency: 0 (write appears next cycle)
// =============================================================================

`timescale 1ns/1ps

module fm_pingpong_buf #(
  parameter int unsigned DEPTH = 786432  // default: 80×80×128 bytes
)(
  input  logic        clock,

  // ---- Bank select ----------------------------------------------------------
  input  logic        bank_sel,    // 0=Bank-A write / B read, 1=B write / A read

  // ---- Write port -----------------------------------------------------------
  input  logic [20:0] wr_addr,     // byte address within one bank (log2(DEPTH))
  input  logic  [7:0] wr_data,
  input  logic        wr_en,

  // ---- Read port ------------------------------------------------------------
  input  logic [20:0] rd_addr,
  output logic  [7:0] rd_data
);

  localparam int unsigned ADDR_W = $clog2(DEPTH);

  // ---------------------------------------------------------------------------
  // Two banks of block RAM
  // ---------------------------------------------------------------------------
  (* ram_style = "block" *)
  logic [7:0] bank_A [0:DEPTH-1];

  (* ram_style = "block" *)
  logic [7:0] bank_B [0:DEPTH-1];

  // Write path
  always_ff @(posedge clock) begin
    if (wr_en) begin
      if (bank_sel == 1'b0)
        bank_A[wr_addr[ADDR_W-1:0]] <= wr_data;
      else
        bank_B[wr_addr[ADDR_W-1:0]] <= wr_data;
    end
  end

  // Read path (1-cycle latency)
  always_ff @(posedge clock) begin
    if (bank_sel == 1'b0)
      rd_data <= bank_B[rd_addr[ADDR_W-1:0]];
    else
      rd_data <= bank_A[rd_addr[ADDR_W-1:0]];
  end

endmodule : fm_pingpong_buf
