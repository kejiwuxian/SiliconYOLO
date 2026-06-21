// =============================================================================
// yolov10n_accel_top.sv  —  YOLOv10n Accelerator Top-Level Integration
// =============================================================================
// Integrates all sub-modules:
//   input_preprocessor  — RGB normalisation + AXI4-Stream framing
//   layer_scheduler     — master FSM + address generation
//   sce_mac_array       — 1024-wide INT8 MAC array (0 DSP)
//   weight_rom_bank     — unified weight + bias BRAM ROMs
//   fm_pingpong_buf     — feature-map double-buffer (IFM/OFM + skip)
//   silu_lut_rom ×64    — per-channel SiLU activation
//   requant_unit        — INT32 → INT8 with bias + shift
//   axi4l_csr           — control/status registers
//
// Skip / concat buffers:
//   skip_buf_p3  — 80×80×64  = 409,600 bytes  (~14 BRAMs)
//   skip_buf_p4  — 40×40×128 = 204,800 bytes  (~7  BRAMs)
//   skip_buf_p5  — 20×20×256 = 102,400 bytes  (~4  BRAMs)
//   skip_buf_p4p — 40×40×128 = 204,800 bytes  (~7  BRAMs, PAN skip)
//
// Main FM buffer DEPTH = 80×80×256 = 1,638,400 bytes (~45 BRAMs per bank).
// With ping-pong that is 90 BRAMs for the main buffer pair.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module yolov10n_accel_top (
  input  logic        clock,
  input  logic        reset,       // active-high synchronous

  // ---- AXI4-Lite CSR (from host) ------------------------------------------
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

  // ---- AXI4-Stream video input (24-bit RGB) --------------------------------
  input  logic [23:0] s_axis_video_tdata,
  input  logic        s_axis_video_tvalid,
  output logic        s_axis_video_tready,
  input  logic        s_axis_video_tuser,   // SOF
  input  logic        s_axis_video_tlast,   // EOL

  // ---- AXI4-Stream detection output (128-bit packed record) ----------------
  output logic [127:0] m_axis_det_tdata,
  output logic         m_axis_det_tvalid,
  input  logic         m_axis_det_tready,
  output logic         m_axis_det_tuser,    // start of output frame
  output logic         m_axis_det_tlast,    // last detection record

  // ---- Interrupt -----------------------------------------------------------
  output logic        irq
);

  // ---------------------------------------------------------------------------
  // Internal signal declarations
  // ---------------------------------------------------------------------------

  // CSR control outputs
  logic        ctrl_start;
  logic        ctrl_flush;
  logic        ctrl_stream_mode;
  logic        ctrl_clk_gate_en;
  logic  [7:0] det_thresh;
  logic [31:0] quant_override;

  // Scheduler control
  logic        sched_busy;
  logic        sched_frame_done;
  logic [21:0] sched_w_addr;
  logic        sched_w_rd_en;
  logic [13:0] sched_b_addr;
  logic        sched_b_rd_en;
  logic        sched_mac_en;
  logic        sched_acc_clear;
  logic        sched_dw_mode;
  logic [20:0] sched_ifm_rd_addr;
  logic        sched_ifm_bank_sel;
  logic [20:0] sched_ofm_wr_addr;
  logic  [7:0] sched_ofm_wr_data;
  logic        sched_ofm_wr_en;
  logic        sched_ofm_bank_sel;
  logic        sched_skip_wr_en;
  logic  [1:0] sched_skip_sel;
  logic [20:0] sched_skip_wr_addr;
  logic  [3:0] sched_shift_bits;
  logic        sched_req_valid_in;
  logic        sched_bias_en;
  logic  [1:0] sched_act_sel;
  logic  [6:0] sched_current_node;
  logic  [3:0] sched_current_stage;

  // Weight ROM
  logic  [7:0] w_data;
  logic [31:0] b_data;

  // SCE weight bus (assembled from sequential ROM reads)
  logic  [7:0] weight_bus [0:P_MAC_WIDTH-1];
  logic [20:0] weight_bus_cnt;

  // SCE accumulator outputs
  logic signed [31:0] acc_out [0:P_COUT_PAR-1];
  logic               acc_valid;

  // Bias bank (one bias per output channel, latched as scheduler iterates)
  logic signed [31:0] bias_bank [0:P_COUT_PAR-1];
  logic               bias_latch_en;

  // Requant outputs
  logic signed  [7:0] y_out   [0:P_COUT_PAR-1];
  logic                reqnt_valid;

  // SiLU outputs
  logic  [7:0] silu_out [0:P_COUT_PAR-1];

  // Main FM ping-pong buffer
  logic  [7:0] ifm_rd_data;
  logic  [7:0] ofm_rd_data;

  // Performance counters
  logic [63:0] perf_cycles_q;
  logic [31:0] latency_last_q;
  logic [31:0] frame_count_q;
  logic [31:0] latency_cnt_q;

  // ---------------------------------------------------------------------------
  // Input Preprocessor
  // ---------------------------------------------------------------------------
  logic  [7:0] pp_tdata;
  logic        pp_tvalid;
  logic        pp_tready;
  logic        pp_tlast;
  logic        pp_tuser;
  logic        pp_sof;
  logic        pp_eof;

  input_preprocessor u_input_prep (
    .clock            (clock),
    .reset            (reset),
    .s_axis_tdata     (s_axis_video_tdata),
    .s_axis_tvalid    (s_axis_video_tvalid),
    .s_axis_tready    (s_axis_video_tready),
    .s_axis_tuser     (s_axis_video_tuser),
    .s_axis_tlast     (s_axis_video_tlast),
    .m_axis_tdata     (pp_tdata),
    .m_axis_tvalid    (pp_tvalid),
    .m_axis_tready    (pp_tready),
    .m_axis_tlast     (pp_tlast),
    .m_axis_tuser     (pp_tuser),
    .sof_pulse        (pp_sof),
    .eof_pulse        (pp_eof)
  );

  // The preprocessed stream writes directly to the main FM buffer
  // (handled as a pre-inference write pass; sched starts after EOF)
  assign pp_tready = 1'b1;  // always accept (buffer has sufficient depth)

  // ---------------------------------------------------------------------------
  // Layer Scheduler
  // ---------------------------------------------------------------------------
  layer_scheduler u_sched (
    .clock             (clock),
    .reset             (reset),
    .start             (ctrl_start),
    .busy              (sched_busy),
    .frame_done        (sched_frame_done),
    .w_addr            (sched_w_addr),
    .w_rd_en           (sched_w_rd_en),
    .b_addr            (sched_b_addr),
    .b_rd_en           (sched_b_rd_en),
    .mac_en            (sched_mac_en),
    .acc_clear         (sched_acc_clear),
    .dw_mode           (sched_dw_mode),
    .weight_bus        (weight_bus),
    .shift_bits        (sched_shift_bits),
    .req_valid_in      (sched_req_valid_in),
    .bias_en           (sched_bias_en),
    .act_sel           (sched_act_sel),
    .ifm_rd_addr       (sched_ifm_rd_addr),
    .ifm_bank_sel      (sched_ifm_bank_sel),
    .ofm_wr_addr       (sched_ofm_wr_addr),
    .ofm_wr_data       (sched_ofm_wr_data),
    .ofm_wr_en         (sched_ofm_wr_en),
    .ofm_bank_sel      (sched_ofm_bank_sel),
    .skip_wr_en        (sched_skip_wr_en),
    .skip_sel          (sched_skip_sel),
    .skip_wr_addr      (sched_skip_wr_addr),
    .current_node      (sched_current_node),
    .current_stage     (sched_current_stage)
  );

  // ---------------------------------------------------------------------------
  // Weight ROM Bank
  // ---------------------------------------------------------------------------
  weight_rom_bank u_weight_rom (
    .clock    (clock),
    .w_addr   (sched_w_addr),
    .w_rd_en  (sched_w_rd_en),
    .w_data   (w_data),
    .b_addr   (sched_b_addr),
    .b_rd_en  (sched_b_rd_en),
    .b_data   (b_data)
  );

  // ---------------------------------------------------------------------------
  // Weight bus assembly
  // Scheduler reads P_MAC_WIDTH bytes sequentially (one per cycle) and
  // assembles them into the weight_bus. weight_bus is latched when
  // weight_bus_cnt == P_MAC_WIDTH - 1.
  // ---------------------------------------------------------------------------
  always_ff @(posedge clock) begin
    if (reset) begin
      weight_bus_cnt <= '0;
    end else if (sched_w_rd_en) begin
      weight_bus[weight_bus_cnt[9:0]] <= w_data;
      if (weight_bus_cnt == P_MAC_WIDTH - 1)
        weight_bus_cnt <= '0;
      else
        weight_bus_cnt <= weight_bus_cnt + 1;
    end
  end

  // ---------------------------------------------------------------------------
  // Bias bank assembly: latch P_COUT_PAR biases when scheduler reads them
  // ---------------------------------------------------------------------------
  logic [5:0] bias_lane_cnt;
  always_ff @(posedge clock) begin
    if (reset) begin
      bias_lane_cnt <= '0;
    end else if (sched_b_rd_en) begin
      bias_bank[bias_lane_cnt] <= b_data;
      if (bias_lane_cnt == P_COUT_PAR - 1)
        bias_lane_cnt <= '0;
      else
        bias_lane_cnt <= bias_lane_cnt + 1;
    end
  end

  // ---------------------------------------------------------------------------
  // SCE MAC Array
  // ---------------------------------------------------------------------------
  sce_mac_array u_sce (
    .clock      (clock),
    .reset      (reset),
    .mac_en     (sched_mac_en),
    .acc_clear  (sched_acc_clear),
    .dw_mode    (sched_dw_mode),
    .weight_bus (weight_bus),
    .act_bus    (weight_bus[0:P_CIN_PAR-1]),  // first CIN_PAR bytes are act
    .acc_out    (acc_out),
    .acc_valid  (acc_valid)
  );
  // NOTE: act_bus should be driven from the IFM buffer read data (P_CIN_PAR
  // bytes wide).  The above is a placeholder; the IFM read path assembles
  // P_CIN_PAR bytes similarly to the weight bus.  See ifm_act_bus below.

  // IFM activation bus (P_CIN_PAR bytes assembled from FM buffer reads)
  logic [7:0] ifm_act_bus [0:P_CIN_PAR-1];
  logic [3:0] ifm_lane_cnt;
  always_ff @(posedge clock) begin
    if (reset) begin
      ifm_lane_cnt <= '0;
    end else begin
      ifm_act_bus[ifm_lane_cnt] <= ifm_rd_data;
      if (sched_mac_en) begin
        if (ifm_lane_cnt == P_CIN_PAR - 1)
          ifm_lane_cnt <= '0;
        else
          ifm_lane_cnt <= ifm_lane_cnt + 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Requantisation Unit
  // ---------------------------------------------------------------------------
  requant_unit u_reqnt (
    .clock      (clock),
    .reset      (reset),
    .shift_bits (sched_shift_bits),
    .valid_in   (sched_req_valid_in),
    .act_type   (sched_act_sel),
    .acc_in     (acc_out),
    .bias_in    (bias_bank),
    .bias_en    (sched_bias_en),
    .y_out      (y_out),
    .valid_out  (reqnt_valid)
  );

  // ---------------------------------------------------------------------------
  // SiLU LUT ROM (P_COUT_PAR instances)
  // ---------------------------------------------------------------------------
  genvar gi;
  generate
    for (gi = 0; gi < P_COUT_PAR; gi++) begin : gen_silu
      silu_lut_rom u_silu (
        .clock  (clock),
        .en     (sched_act_sel == ACT_SILU),
        .x_in   (y_out[gi]),
        .y_out  (silu_out[gi])
      );
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Main Feature-Map Ping-Pong Buffer
  // (80×80×256 = 1,638,400 bytes per bank)
  // ---------------------------------------------------------------------------
  fm_pingpong_buf #(.DEPTH(1638400)) u_fm_main (
    .clock      (clock),
    .bank_sel   (sched_ofm_bank_sel),
    .wr_addr    (sched_ofm_wr_addr),
    .wr_data    (sched_ofm_wr_data),
    .wr_en      (sched_ofm_wr_en),
    .rd_addr    (sched_ifm_rd_addr),
    .rd_data    (ifm_rd_data)
  );

  // ---------------------------------------------------------------------------
  // Skip Buffers (P3, P4, P5, P4_pan)
  // ---------------------------------------------------------------------------
  logic [7:0] skip_wr_data;
  logic [7:0] skip_rd_data [0:3];

  assign skip_wr_data = silu_out[0];  // placeholder: driven from OFM output path

  fm_pingpong_buf #(.DEPTH(409600))  u_skip_p3  (.clock(clock),.bank_sel(sched_skip_sel==2'd0 ? 1'b0 : 1'b1),.wr_addr(sched_skip_wr_addr[18:0]),.wr_data(skip_wr_data),.wr_en(sched_skip_wr_en && sched_skip_sel==2'd0),.rd_addr(sched_ifm_rd_addr[18:0]),.rd_data(skip_rd_data[0]));
  fm_pingpong_buf #(.DEPTH(204800))  u_skip_p4  (.clock(clock),.bank_sel(sched_skip_sel==2'd1 ? 1'b0 : 1'b1),.wr_addr(sched_skip_wr_addr[17:0]),.wr_data(skip_wr_data),.wr_en(sched_skip_wr_en && sched_skip_sel==2'd1),.rd_addr(sched_ifm_rd_addr[17:0]),.rd_data(skip_rd_data[1]));
  fm_pingpong_buf #(.DEPTH(102400))  u_skip_p5  (.clock(clock),.bank_sel(sched_skip_sel==2'd2 ? 1'b0 : 1'b1),.wr_addr(sched_skip_wr_addr[16:0]),.wr_data(skip_wr_data),.wr_en(sched_skip_wr_en && sched_skip_sel==2'd2),.rd_addr(sched_ifm_rd_addr[16:0]),.rd_data(skip_rd_data[2]));
  fm_pingpong_buf #(.DEPTH(204800))  u_skip_p4p (.clock(clock),.bank_sel(sched_skip_sel==2'd3 ? 1'b0 : 1'b1),.wr_addr(sched_skip_wr_addr[17:0]),.wr_data(skip_wr_data),.wr_en(sched_skip_wr_en && sched_skip_sel==2'd3),.rd_addr(sched_ifm_rd_addr[17:0]),.rd_data(skip_rd_data[3]));

  // ---------------------------------------------------------------------------
  // Performance counters
  // ---------------------------------------------------------------------------
  always_ff @(posedge clock) begin
    if (reset) begin
      perf_cycles_q  <= '0;
      latency_cnt_q  <= '0;
      latency_last_q <= '0;
      frame_count_q  <= '0;
    end else begin
      perf_cycles_q <= perf_cycles_q + 1;
      if (sched_busy)
        latency_cnt_q <= latency_cnt_q + 1;
      if (sched_frame_done) begin
        latency_last_q <= latency_cnt_q;
        latency_cnt_q  <= '0;
        frame_count_q  <= frame_count_q + 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Detection output (stub: passes head outputs from ofm buffer)
  // In final integration, decode the last 6 layers' OFM into packed records.
  // ---------------------------------------------------------------------------
  assign m_axis_det_tdata  = '0;  // TODO: connect head output decoder
  assign m_axis_det_tvalid = sched_frame_done;
  assign m_axis_det_tuser  = sched_frame_done;
  assign m_axis_det_tlast  = sched_frame_done;

  // ---------------------------------------------------------------------------
  // AXI4-Lite CSR
  // ---------------------------------------------------------------------------
  axi4l_csr u_csr (
    .clock              (clock),
    .reset              (reset),
    .s_axil_awaddr      (s_axil_awaddr),
    .s_axil_awvalid     (s_axil_awvalid),
    .s_axil_awready     (s_axil_awready),
    .s_axil_wdata       (s_axil_wdata),
    .s_axil_wstrb       (s_axil_wstrb),
    .s_axil_wvalid      (s_axil_wvalid),
    .s_axil_wready      (s_axil_wready),
    .s_axil_bresp       (s_axil_bresp),
    .s_axil_bvalid      (s_axil_bvalid),
    .s_axil_bready      (s_axil_bready),
    .s_axil_araddr      (s_axil_araddr),
    .s_axil_arvalid     (s_axil_arvalid),
    .s_axil_arready     (s_axil_arready),
    .s_axil_rdata       (s_axil_rdata),
    .s_axil_rresp       (s_axil_rresp),
    .s_axil_rvalid      (s_axil_rvalid),
    .s_axil_rready      (s_axil_rready),
    .hw_busy            (sched_busy),
    .hw_frame_done      (sched_frame_done),
    .hw_error           (1'b0),
    .hw_stage_id        (sched_current_stage),
    .hw_output_valid    (sched_frame_done),
    .hw_overflow        (1'b0),
    .hw_det_count       (16'd0),
    .hw_perf_cycles     (perf_cycles_q),
    .hw_latency_last    (latency_last_q),
    .hw_frame_count     (frame_count_q),
    .ctrl_start         (ctrl_start),
    .ctrl_flush         (ctrl_flush),
    .ctrl_stream_mode   (ctrl_stream_mode),
    .ctrl_clk_gate_en   (ctrl_clk_gate_en),
    .det_thresh         (det_thresh),
    .quant_override     (quant_override),
    .irq                (irq)
  );

endmodule : yolov10n_accel_top
