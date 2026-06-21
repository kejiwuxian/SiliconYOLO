// =============================================================================
// input_preprocessor.sv  —  AXI4-Stream RGB -> INT8 Normalisation
// =============================================================================
// Accepts 24-bit RGB pixels on s_axis_video (AXI4-Stream, TUSER=SOF, TLAST=EOL)
// and converts them to INT8 using the YOLOv10n PTQ input normalisation:
//
//   x_int8 = clip(round((pixel / 255.0 - 0.0) / act_scale_in_layer0), -127, 127)
//
// From quant_scales.json layer 0: act_scale_in = 0.007874016  (= 1/127)
//   => x_int8 = clip(round(pixel * 127 / 255), 0, 127)
//            ≈ clip(round(pixel / 2.008), 0, 127)
//   which is simply: x_int8 = pixel[7:1] (divide by 2, truncate)
//   A more precise approximation: x_int8 = (pixel * 127 + 127) >> 8
//
// Outputs:
//   m_axis_pp  — AXI4-Stream of INT8 values, same framing as input
//   sof_out    — start-of-frame pulse (used by scheduler to latch frame start)
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module input_preprocessor (
  input  logic        clock,
  input  logic        reset,

  // ---- AXI4-Stream input (24-bit RGB, TUSER=SOF, TLAST=EOL) ---------------
  input  logic [23:0] s_axis_tdata,
  input  logic        s_axis_tvalid,
  output logic        s_axis_tready,
  input  logic        s_axis_tuser,   // SOF
  input  logic        s_axis_tlast,   // EOL

  // ---- Preprocessed output (3 × INT8 per pixel, serialised as R,G,B) ------
  output logic  [7:0] m_axis_tdata,   // one INT8 channel per beat
  output logic        m_axis_tvalid,
  input  logic        m_axis_tready,
  output logic        m_axis_tlast,
  output logic        m_axis_tuser,   // SOF on R channel of first pixel

  // ---- Frame control -------------------------------------------------------
  output logic        sof_pulse,      // single-cycle start-of-frame pulse
  output logic        eof_pulse       // single-cycle end-of-frame pulse
);

  // ---------------------------------------------------------------------------
  // Normalise one 8-bit channel: x_int8 = (pixel * 127 + 127) >> 8
  // This maps [0,255] -> [0,127] which is the correct range for layer 0.
  // ---------------------------------------------------------------------------
  function automatic logic signed [7:0] norm8 (input logic [7:0] pixel);
    automatic logic [15:0] tmp;
    tmp = (pixel * 8'd127 + 8'd127);
    norm8 = tmp[15:8];  // >> 8
  endfunction

  // ---------------------------------------------------------------------------
  // Demux RGB -> R, G, B as 3 sequential INT8 beats
  // ---------------------------------------------------------------------------
  logic [7:0] r_ch, g_ch, b_ch;
  logic [1:0] ch_cnt;    // 0=R, 1=G, 2=B
  logic       pkt_valid;

  logic [23:0] hold_pix;
  logic        hold_valid;
  logic        hold_last;
  logic        hold_user;

  // Accept one pixel per 3-beat burst
  assign s_axis_tready = (ch_cnt == 2'd2) && m_axis_tready;

  always_comb begin
    r_ch = norm8(hold_pix[23:16]);
    g_ch = norm8(hold_pix[15:8]);
    b_ch = norm8(hold_pix[7:0]);
  end

  always_ff @(posedge clock) begin
    if (reset) begin
      ch_cnt    <= 2'd0;
      hold_valid <= 1'b0;
      sof_pulse <= 1'b0;
      eof_pulse <= 1'b0;
    end else begin
      sof_pulse <= 1'b0;
      eof_pulse <= 1'b0;

      if (s_axis_tvalid && s_axis_tready) begin
        hold_pix   <= s_axis_tdata;
        hold_valid <= 1'b1;
        hold_last  <= s_axis_tlast;
        hold_user  <= s_axis_tuser;
        ch_cnt     <= 2'd0;
        if (s_axis_tuser) sof_pulse <= 1'b1;
        if (s_axis_tlast) eof_pulse <= 1'b1;
      end else if (hold_valid && m_axis_tready) begin
        if (ch_cnt == 2'd2)
          hold_valid <= 1'b0;
        else
          ch_cnt <= ch_cnt + 1;
      end
    end
  end

  // Output mux
  always_comb begin
    case (ch_cnt)
      2'd0:    m_axis_tdata = r_ch;
      2'd1:    m_axis_tdata = g_ch;
      2'd2:    m_axis_tdata = b_ch;
      default: m_axis_tdata = '0;
    endcase
    m_axis_tvalid = hold_valid;
    m_axis_tlast  = hold_last && (ch_cnt == 2'd2);
    m_axis_tuser  = hold_user && (ch_cnt == 2'd0);
  end

endmodule : input_preprocessor
