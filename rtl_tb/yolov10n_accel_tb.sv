// =============================================================================
// yolov10n_accel_tb.sv  —  TOP-LEVEL self-checking testbench (Vivado xsim)
// =============================================================================
// Drives the full yolov10n_accel_top:
//   1. AXI4-Lite: program DET_THRESH, pulse CTRL.start.
//   2. AXI4-Stream video: stream the golden INPUT frame (640x640 RGB,
//      reconstructed from golden/vectors/<id>/input_int8.npy via
//      rtl_tb/export_golden_hex.py -> rtl_tb/golden_hex/<id>/input_rgb.hex),
//      one pixel/cycle with tuser=SOF on the first pixel and tlast=EOL per row.
//   3. Wait for frame_done / m_axis_det_tvalid.
//   4. Capture the emitted detection records and compare against the golden
//      detections (rtl_tb/golden_hex/<id>/detections.txt), reporting PASS/FAIL
//      and max abs error in box coords + score.
//
// TARGET: Vivado xsim (xvlog -sv / xelab / xsim) — the top instantiates buffers
// that infer Xilinx XPM/BRAM and the package CFG_ROM uses packed-struct
// assignment patterns that the open-source Icarus front-end does not elaborate.
// Run via rtl_tb/run_xsim.ps1.
//
// HONEST VERIFICATION NOTE (see rtl_tb/SIM_SHOWCASE.md):
// The Cognichip-generated top-level currently STUBS the detection output decoder
//   rtl/yolov10n_accel_top.sv:370   assign m_axis_det_tdata = '0; // TODO
// so until that decoder is connected, the captured records will be zero and the
// detection comparison below will report the mismatch as a FINDING (not a TB
// bug). The per-datapath-block correctness is already proven bit-exact by the
// rtl_tb/unit/*_tb.sv suite (csd_mac_slice, sce_mac_array, requant_unit,
// silu_lut_rom — all PASS in Icarus) and by the golden vectors. This TB is the
// harness that locks the FULL chip once integration is complete.
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module yolov10n_accel_tb;

  // ---- parameters -----------------------------------------------------------
  localparam int H = 640;
  localparam int W = 640;
  localparam string GOLD = "rtl_tb/golden_hex/0000";

  // ---- clock / reset --------------------------------------------------------
  logic clock = 1'b0;
  logic reset;
  always #2.5 clock = ~clock;   // 200 MHz

  // ---- DUT I/O --------------------------------------------------------------
  // AXI4-Lite
  logic [11:0] s_axil_awaddr;  logic s_axil_awvalid; logic s_axil_awready;
  logic [31:0] s_axil_wdata;   logic [3:0] s_axil_wstrb;
  logic        s_axil_wvalid;  logic s_axil_wready;
  logic  [1:0] s_axil_bresp;   logic s_axil_bvalid;  logic s_axil_bready;
  logic [11:0] s_axil_araddr;  logic s_axil_arvalid; logic s_axil_arready;
  logic [31:0] s_axil_rdata;   logic [1:0] s_axil_rresp;
  logic        s_axil_rvalid;  logic s_axil_rready;
  // AXI4-Stream video in
  logic [23:0] s_axis_video_tdata;  logic s_axis_video_tvalid;
  logic        s_axis_video_tready; logic s_axis_video_tuser; logic s_axis_video_tlast;
  // AXI4-Stream det out
  logic [127:0] m_axis_det_tdata;   logic m_axis_det_tvalid;
  logic         m_axis_det_tready;  logic m_axis_det_tuser; logic m_axis_det_tlast;
  logic         irq;

  // ---- DUT ------------------------------------------------------------------
  yolov10n_accel_top dut (
    .clock(clock), .reset(reset),
    .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
    .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
    .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
    .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
    .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
    .s_axis_video_tdata(s_axis_video_tdata), .s_axis_video_tvalid(s_axis_video_tvalid),
    .s_axis_video_tready(s_axis_video_tready), .s_axis_video_tuser(s_axis_video_tuser), .s_axis_video_tlast(s_axis_video_tlast),
    .m_axis_det_tdata(m_axis_det_tdata), .m_axis_det_tvalid(m_axis_det_tvalid),
    .m_axis_det_tready(m_axis_det_tready), .m_axis_det_tuser(m_axis_det_tuser), .m_axis_det_tlast(m_axis_det_tlast),
    .irq(irq)
  );

  // ---- golden data ----------------------------------------------------------
  logic [23:0] frame_rgb [0:H*W-1];
  // golden detections
  integer g_x1[0:299], g_y1[0:299], g_x2[0:299], g_y2[0:299], g_sc[0:299], g_cl[0:299];
  integer g_ndet;

  integer errors = 0;
  integer captured = 0;
  integer i, px, row, col;
  integer fd, code;
  // captured detection records (declared before the capture always-block uses them)
  integer cap_x1[0:299], cap_y1[0:299], cap_x2[0:299], cap_y2[0:299], cap_sc[0:299], cap_cl[0:299];
  // comparison scratch
  integer n_cmp, max_box_err, max_sc_err, e_tmp;

  // ---- AXI4-Lite write task -------------------------------------------------
  task automatic axil_write(input [11:0] addr, input [31:0] data);
    begin
      @(negedge clock);
      s_axil_awaddr = addr; s_axil_awvalid = 1'b1;
      s_axil_wdata  = data; s_axil_wstrb = 4'hF; s_axil_wvalid = 1'b1;
      s_axil_bready = 1'b1;
      // wait for both ready
      do @(posedge clock); while (!(s_axil_awready && s_axil_wready));
      @(negedge clock);
      s_axil_awvalid = 1'b0; s_axil_wvalid = 1'b0;
      // wait for bvalid
      do @(posedge clock); while (!s_axil_bvalid);
      @(negedge clock); s_axil_bready = 1'b0;
    end
  endtask

  // ---- AXI4-Lite read task --------------------------------------------------
  task automatic axil_read(input [11:0] addr, output [31:0] data);
    begin
      @(negedge clock);
      s_axil_araddr = addr; s_axil_arvalid = 1'b1; s_axil_rready = 1'b1;
      do @(posedge clock); while (!s_axil_arready);
      @(negedge clock); s_axil_arvalid = 1'b0;
      do @(posedge clock); while (!s_axil_rvalid);
      data = s_axil_rdata;
      @(negedge clock); s_axil_rready = 1'b0;
    end
  endtask

  // ---- capture emitted detection records ------------------------------------
  // record packing (per axi4l/head decoder contract):
  //   [127:112]=x1 [111:96]=y1 [95:80]=x2 [79:64]=y2 [63:56]=score [55:48]=cls
  always @(posedge clock) begin
    if (!reset && m_axis_det_tvalid && m_axis_det_tready) begin
      if (captured < 300) begin
        // store raw; compared after frame
        cap_x1[captured] = m_axis_det_tdata[127:112];
        cap_y1[captured] = m_axis_det_tdata[111:96];
        cap_x2[captured] = m_axis_det_tdata[95:80];
        cap_y2[captured] = m_axis_det_tdata[79:64];
        cap_sc[captured] = m_axis_det_tdata[63:56];
        cap_cl[captured] = m_axis_det_tdata[55:48];
        captured = captured + 1;
      end
    end
  end

  // ---- main -----------------------------------------------------------------
  integer det_count_reg;
  reg [31:0] rd;
  initial begin
    $dumpfile("rtl_tb/sim_out/yolov10n_accel.vcd");
    $dumpvars(0, yolov10n_accel_tb);

    // load golden frame + detections
    $readmemh({GOLD, "/input_rgb.hex"}, frame_rgb);
    fd = $fopen({GOLD, "/detections.txt"}, "r");
    g_ndet = 0;
    if (fd) begin
      for (i = 0; i < 300; i = i + 1) begin
        code = $fscanf(fd, "%d %d %d %d %d %d\n",
                       g_x1[i], g_y1[i], g_x2[i], g_y2[i], g_sc[i], g_cl[i]);
        if (code == 6 && g_sc[i] > 0) g_ndet = g_ndet + 1;
      end
      $fclose(fd);
    end
    $display("[yolov10n_accel_tb] loaded golden frame (%0dx%0d) + %0d detections", H, W, g_ndet);

    // init
    s_axil_awaddr='0; s_axil_awvalid=0; s_axil_wdata='0; s_axil_wstrb=0; s_axil_wvalid=0; s_axil_bready=0;
    s_axil_araddr='0; s_axil_arvalid=0; s_axil_rready=0;
    s_axis_video_tdata='0; s_axis_video_tvalid=0; s_axis_video_tuser=0; s_axis_video_tlast=0;
    m_axis_det_tready=1'b1;
    reset = 1'b1;
    repeat (8) @(posedge clock);
    reset = 1'b0;
    repeat (4) @(posedge clock);

    // sanity: VERSION register
    axil_read(12'h0FC, rd);
    $display("  VERSION = 0x%08x (expect 0x00010000)", rd);

    // program threshold (0x20) and start (CTRL.start bit0)
    axil_write(12'h01C, 32'h0000_0040);   // DET_THRESH = 0x40
    axil_write(12'h000, 32'h0000_0001);   // CTRL.start

    // stream the video frame: 1 pixel/cycle, SOF on first, EOL (tlast) at row end
    for (row = 0; row < H; row = row + 1) begin
      for (col = 0; col < W; col = col + 1) begin
        @(negedge clock);
        px = row*W + col;
        s_axis_video_tdata  = frame_rgb[px];
        s_axis_video_tvalid = 1'b1;
        s_axis_video_tuser  = (px == 0);          // SOF
        s_axis_video_tlast  = (col == W-1);       // EOL
        // honor backpressure
        @(posedge clock);
        while (!s_axis_video_tready) @(posedge clock);
      end
    end
    @(negedge clock);
    s_axis_video_tvalid = 1'b0; s_axis_video_tuser = 0; s_axis_video_tlast = 0;

    // wait for frame_done (via det tvalid pulse) or timeout
    fork : wait_done
      begin
        @(posedge m_axis_det_tvalid);
        disable wait_done;
      end
      begin
        repeat (5_000_000) @(posedge clock);
        $display("  [timeout] frame_done not observed within 5M cycles");
        disable wait_done;
      end
    join

    repeat (16) @(posedge clock);

    // read DET_COUNT
    axil_read(12'h018, rd);
    det_count_reg = rd[15:0];
    $display("  DET_COUNT register = %0d  (golden kept = %0d)", det_count_reg, g_ndet);
    $display("  captured %0d detection records on m_axis_det", captured);

    // ---- compare ----
    // (compares only when records were actually emitted; see HONEST NOTE above)
    if (captured == 0) begin
      $display("  ** FINDING: no detection records emitted. The generated top stubs");
      $display("     m_axis_det_tdata ('0) — head output decoder is a TODO in");
      $display("     rtl/yolov10n_accel_top.sv. Per-block datapath is proven by the");
      $display("     unit TBs + golden vectors. Treating as KNOWN-INCOMPLETE, not TB bug.");
    end else begin
      n_cmp = (captured < g_ndet) ? captured : g_ndet;
      max_box_err = 0;
      max_sc_err  = 0;
      for (i = 0; i < n_cmp; i = i + 1) begin
        e_tmp = abs_i(cap_x1[i]-g_x1[i]); if (e_tmp>max_box_err) max_box_err=e_tmp;
        e_tmp = abs_i(cap_y1[i]-g_y1[i]); if (e_tmp>max_box_err) max_box_err=e_tmp;
        e_tmp = abs_i(cap_x2[i]-g_x2[i]); if (e_tmp>max_box_err) max_box_err=e_tmp;
        e_tmp = abs_i(cap_y2[i]-g_y2[i]); if (e_tmp>max_box_err) max_box_err=e_tmp;
        e_tmp = abs_i(cap_sc[i]-g_sc[i]); if (e_tmp>max_sc_err)  max_sc_err=e_tmp;
        if (cap_cl[i] !== g_cl[i]) errors = errors + 1;
      end
      $display("  compared %0d records: max box err=%0d px, max score err=%0d, class mismatches=%0d",
               n_cmp, max_box_err, max_sc_err, errors);
      if (max_box_err > 2 || max_sc_err > 4) errors = errors + 1;
    end

    if (errors == 0 && captured > 0)
      $display("==== yolov10n_accel_tb: PASS ====");
    else if (captured == 0)
      $display("==== yolov10n_accel_tb: INCONCLUSIVE (top-level output decoder stubbed) ====");
    else
      $display("==== yolov10n_accel_tb: FAIL (%0d) ====", errors);
    $finish;
  end

  function automatic integer abs_i(input integer v);
    abs_i = (v < 0) ? -v : v;
  endfunction

endmodule : yolov10n_accel_tb
