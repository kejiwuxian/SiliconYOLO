// =============================================================================
// axi4l_csr.sv  —  AXI4-Lite Control / Status Register Block
// =============================================================================
// Implements the register map from yolov10n_accel_spec.md §6.
// Base offset: 0x000 (top-level maps to 0x4000_0000 on Genesys 2).
//
// Register map:
//   0x000 CTRL         RW  {flush,clk_gate_en,stream_mode,start}
//   0x004 STATUS       RO  {stage_id[3:0],output_valid,overflow,error,busy}
//   0x008 IRQ_STATUS   W1C {flush_done,error,frame_done}
//   0x00C IRQ_ENABLE   RW  {flush_done,error,frame_done}
//   0x010 FRAME_COUNT  RO  frames since reset
//   0x014 LATENCY_LAST RO  cycles for last frame
//   0x018 DET_COUNT    RO  detections in last frame [15:0]
//   0x01C DET_THRESH   RW  score threshold [7:0]
//   0x020 QUANT_OVERRIDE RW INT4 layer override bitmask
//   0x024 PERF_LO      RO  total cycles [31:0]
//   0x028 PERF_HI      RO  total cycles [63:32]
//   0x0FC VERSION      RO  0x0001_0000
// =============================================================================

`timescale 1ns/1ps

module axi4l_csr (
  input  logic        clock,
  input  logic        reset,

  // ---- AXI4-Lite slave ----------------------------------------------------
  input  logic [11:0] s_axil_awaddr,
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,
  input  logic  [3:0] s_axil_wstrb,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  output logic  [1:0] s_axil_bresp,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [11:0] s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,
  output logic  [1:0] s_axil_rresp,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,

  // ---- HW status inputs (from accelerator datapath) -----------------------
  input  logic        hw_busy,
  input  logic        hw_frame_done,  // pulse
  input  logic        hw_error,       // pulse
  input  logic  [3:0] hw_stage_id,
  input  logic        hw_output_valid,
  input  logic        hw_overflow,
  input  logic [15:0] hw_det_count,
  input  logic [63:0] hw_perf_cycles,
  input  logic [31:0] hw_latency_last,
  input  logic [31:0] hw_frame_count,

  // ---- SW control outputs (to accelerator) --------------------------------
  output logic        ctrl_start,       // pulse
  output logic        ctrl_flush,       // pulse
  output logic        ctrl_stream_mode,
  output logic        ctrl_clk_gate_en,
  output logic  [7:0] det_thresh,
  output logic [31:0] quant_override,

  // ---- Interrupt -----------------------------------------------------------
  output logic        irq
);

  // ---------------------------------------------------------------------------
  // Register storage
  // ---------------------------------------------------------------------------
  logic [31:0] reg_ctrl;
  logic [31:0] reg_irq_status;
  logic [31:0] reg_irq_enable;
  logic  [7:0] reg_det_thresh;
  logic [31:0] reg_quant_override;

  // Derived status
  logic [31:0] status_rd;
  assign status_rd = {20'b0, hw_stage_id, hw_output_valid, hw_overflow,
                      hw_error, hw_busy};

  // ---------------------------------------------------------------------------
  // AXI4-Lite write channel (simple single-cycle ready)
  // ---------------------------------------------------------------------------
  logic        aw_active;
  logic [11:0] aw_addr_q;

  always_ff @(posedge clock) begin
    if (reset) begin
      s_axil_awready  <= 1'b1;
      s_axil_wready   <= 1'b1;
      s_axil_bvalid   <= 1'b0;
      s_axil_bresp    <= 2'b00;
      aw_active       <= 1'b0;
      reg_ctrl        <= '0;
      reg_irq_enable  <= '0;
      reg_det_thresh  <= 8'h40;
      reg_quant_override <= '0;
      ctrl_start      <= 1'b0;
      ctrl_flush      <= 1'b0;
    end else begin
      ctrl_start <= 1'b0;
      ctrl_flush <= 1'b0;

      // Latch write address
      if (s_axil_awvalid && s_axil_awready) begin
        aw_addr_q  <= s_axil_awaddr;
        aw_active  <= 1'b1;
      end

      // Write data
      if (s_axil_wvalid && s_axil_wready && aw_active) begin
        aw_active     <= 1'b0;
        s_axil_bvalid <= 1'b1;
        case (aw_addr_q[7:0])
          8'h00: begin
            if (s_axil_wstrb[0]) reg_ctrl[7:0]   <= s_axil_wdata[7:0];
            ctrl_start <= s_axil_wdata[0];
            ctrl_flush <= s_axil_wdata[3];
          end
          8'h08: begin
            // W1C: clear bits where wdata=1
            if (s_axil_wstrb[0]) reg_irq_status[7:0] <=
              reg_irq_status[7:0] & ~s_axil_wdata[7:0];
          end
          8'h0C: begin
            if (s_axil_wstrb[0]) reg_irq_enable[7:0] <= s_axil_wdata[7:0];
          end
          8'h1C: begin
            if (s_axil_wstrb[0]) reg_det_thresh <= s_axil_wdata[7:0];
          end
          8'h20: begin
            for (int b = 0; b < 4; b++)
              if (s_axil_wstrb[b]) reg_quant_override[b*8 +: 8] <=
                s_axil_wdata[b*8 +: 8];
          end
          default: ;
        endcase
      end

      // Clear bvalid once accepted
      if (s_axil_bvalid && s_axil_bready)
        s_axil_bvalid <= 1'b0;

      // IRQ status capture
      if (hw_frame_done) reg_irq_status[0] <= 1'b1;
      if (hw_error)      reg_irq_status[1] <= 1'b1;
      // flush_done would be added similarly
    end
  end

  assign ctrl_stream_mode  = reg_ctrl[1];
  assign ctrl_clk_gate_en  = reg_ctrl[2];
  assign det_thresh        = reg_det_thresh;
  assign quant_override    = reg_quant_override;

  // ---------------------------------------------------------------------------
  // AXI4-Lite read channel
  // ---------------------------------------------------------------------------
  always_ff @(posedge clock) begin
    if (reset) begin
      s_axil_arready <= 1'b1;
      s_axil_rvalid  <= 1'b0;
      s_axil_rresp   <= 2'b00;
      s_axil_rdata   <= '0;
    end else begin
      if (s_axil_arvalid && s_axil_arready) begin
        s_axil_rvalid <= 1'b1;
        case (s_axil_araddr[7:0])
          8'h00:  s_axil_rdata <= reg_ctrl;
          8'h04:  s_axil_rdata <= status_rd;
          8'h08:  s_axil_rdata <= reg_irq_status;
          8'h0C:  s_axil_rdata <= reg_irq_enable;
          8'h10:  s_axil_rdata <= hw_frame_count;
          8'h14:  s_axil_rdata <= hw_latency_last;
          8'h18:  s_axil_rdata <= {16'b0, hw_det_count};
          8'h1C:  s_axil_rdata <= {24'b0, reg_det_thresh};
          8'h20:  s_axil_rdata <= reg_quant_override;
          8'h24:  s_axil_rdata <= hw_perf_cycles[31:0];
          8'h28:  s_axil_rdata <= hw_perf_cycles[63:32];
          8'hFC:  s_axil_rdata <= 32'h0001_0000;  // VERSION
          default:s_axil_rdata <= 32'hDEAD_BEEF;
        endcase
      end
      if (s_axil_rvalid && s_axil_rready)
        s_axil_rvalid <= 1'b0;
    end
  end

  // ---------------------------------------------------------------------------
  // Interrupt
  // ---------------------------------------------------------------------------
  assign irq = |(reg_irq_status & reg_irq_enable);

endmodule : axi4l_csr
