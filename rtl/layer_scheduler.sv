// =============================================================================
// layer_scheduler.sv  —  Master Execution FSM + Address Generator
// =============================================================================
// Controls the entire YOLOv10n inference graph.  On each frame it steps
// through all N_NODES (88) compute nodes in execution order, driving:
//   • Weight ROM read addresses + burst count
//   • Feature-map buffer read/write addresses and bank-select
//   • SCE enable, acc_clear, dw_mode signals
//   • Bias + requant control (shift_bits, bias_en)
//   • MaxPool and Upsample passthrough control
//   • Concat topology (hardcoded per YOLOv10n graph)
//
// Concat topology (hardcoded):
//   Node 28 (model.9.cv2):  IFM = cat(sppf_in @ 128ch, pool1,2,3 @ 128ch ea.) = 512ch
//   Node 36 (model.13.cv1): IFM = cat(upsample(node35) @ 256ch, save_p4 @ 128ch) = 384ch
//   Node 40 (model.16.cv1): IFM = cat(upsample(node39) @ 128ch, save_p3 @ 64ch)  = 192ch
//   Node 45 (model.19.cv1): IFM = cat(node44 @ 64ch,   save_p4_pan @ 128ch)      = 192ch
//   Node 51 (model.22.cv1): IFM = cat(node50 @ 128ch,  save_p5 @ 256ch)          = 384ch
//
// Key save points:
//   After node 12 (model.4.cv2)  → save P3 skip (64ch @80x80)
//   After node 20 (model.6.cv2)  → save P4 skip (128ch @40x40)
//   After node 35 (model.10.cv2) → save P5 (256ch @20x20)
//   After node 39 (model.13.cv2) → save P4_pan skip (128ch @40x40)
// =============================================================================

`timescale 1ns/1ps
import yolov10n_pkg::*;

module layer_scheduler (
  input  logic        clock,
  input  logic        reset,

  // ---- Top-level control ---------------------------------------------------
  input  logic        start,         // pulse: begin inference on next cycle
  output logic        busy,          // 1 while inference in progress
  output logic        frame_done,    // pulse: all nodes complete

  // ---- Weight ROM control --------------------------------------------------
  output logic [21:0] w_addr,
  output logic        w_rd_en,
  output logic [13:0] b_addr,
  output logic        b_rd_en,

  // ---- SCE control ---------------------------------------------------------
  output logic        mac_en,
  output logic        acc_clear,
  output logic        dw_mode,
  output logic [7:0]  weight_bus [0:P_MAC_WIDTH-1],  // latched from ROM

  // ---- Requant control -----------------------------------------------------
  output logic [3:0]  shift_bits,
  output logic        req_valid_in,
  output logic        bias_en,
  output logic [1:0]  act_sel,       // ACT_* enum value

  // ---- FM buffer control ---------------------------------------------------
  // IFM (input to SCE)
  output logic [20:0] ifm_rd_addr,
  output logic        ifm_bank_sel,
  // OFM (output from requant)
  output logic [20:0] ofm_wr_addr,
  output logic  [7:0] ofm_wr_data,
  output logic        ofm_wr_en,
  output logic        ofm_bank_sel,

  // ---- Skip-buffer control (P3/P4/P5 lateral connections) -----------------
  output logic        skip_wr_en,
  output logic  [1:0] skip_sel,      // which skip buffer: 0=P3,1=P4,2=P5,3=P4_pan
  output logic [20:0] skip_wr_addr,

  // ---- Status --------------------------------------------------------------
  output logic  [6:0] current_node,  // 0-87 for debug
  output logic  [3:0] current_stage  // pipeline stage for CSR
);

  // ---------------------------------------------------------------------------
  // State machine
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    ST_IDLE         = 4'd0,
    ST_LOAD_CFG     = 4'd1,   // register layer config from CFG_ROM
    ST_PREFETCH_W   = 4'd2,   // pre-fetch first weight row
    ST_MAC_LOOP     = 4'd3,   // inner MAC loop
    ST_BIAS_REQ     = 4'd4,   // fetch bias + trigger requant
    ST_WRITE_OFM    = 4'd5,   // write requantised output to OFM buffer
    ST_ACTIVATE     = 4'd6,   // optional SiLU cycle
    ST_MAXPOOL      = 4'd7,   // 3x3 max-pool pass
    ST_UPSAMPLE     = 4'd8,   // nearest-neighbour 2x upsample
    ST_CONCAT       = 4'd9,   // concat re-map (address-only)
    ST_NEXT_NODE    = 4'd10,  // advance to next node
    ST_FRAME_DONE   = 4'd11   // pulse frame_done
  } state_t;

  state_t state_q, state_d;

  // ---------------------------------------------------------------------------
  // Node / loop counters
  // ---------------------------------------------------------------------------
  logic [6:0]  node_q;       // current node index 0-87
  logic [9:0]  out_row_q;    // output row counter
  logic [9:0]  out_col_q;    // output column counter
  logic [7:0]  cout_grp_q;   // output channel group (0..C_out/64-1)
  logic [5:0]  cin_chk_q;    // input channel chunk  (0..C_in/16-1)
  logic [3:0]  kh_q, kw_q;   // kernel position counters
  logic [3:0]  mac_cnt_q;    // cycles within one (cin_chunk, kh, kw)

  // Current layer configuration (latched at ST_LOAD_CFG)
  layer_cfg_t  cfg_q;

  // Computed derived fields
  logic [7:0]  n_cout_grps;  // ceil(c_out / P_COUT_PAR)
  logic [5:0]  n_cin_chks;   // ceil(c_in  / P_CIN_PAR)
  logic [3:0]  n_ksteps;     // kernel_size

  assign n_cout_grps = (cfg_q.c_out + P_COUT_PAR - 1) / P_COUT_PAR;
  assign n_cin_chks  = cfg_q.depthwise ? 6'd1 :
                       (cfg_q.c_in + P_CIN_PAR - 1) / P_CIN_PAR;
  assign n_ksteps    = cfg_q.kernel;

  // ---------------------------------------------------------------------------
  // Weight fetch address generation
  // ---------------------------------------------------------------------------
  // For each (cout_grp, cin_chk, kh, kw) cycle we need to read
  // P_COUT_PAR × P_CIN_PAR = 1024 bytes from the weight ROM.
  // We read them sequentially in P_CIN_PAR=16 bytes per sub-cycle.
  // (In a real impl with wider BRAM bus this collapses to 1 cycle;
  //  here we model 16 sub-reads per MAC cycle for clarity.)
  //
  // w_base = cfg_q.w_offset + (cout_grp * n_cin_chks * k * k +
  //                             cin_chk  * k * k +
  //                             kh * k + kw) * P_COUT_PAR * P_CIN_PAR
  //        (for DW: per-channel weight at dw_weight_offset)

  logic [21:0] w_base;
  logic [21:0] w_addr_next;

  always_comb begin
    if (cfg_q.depthwise) begin
      // DW: each c_out has 1 × k² weights; weight index = cout_grp*64 * k² + kh*k + kw
      w_base = cfg_q.w_offset +
               (cout_grp_q * n_ksteps * n_ksteps +
                kh_q * n_ksteps + kw_q) * P_COUT_PAR;
    end else begin
      w_base = cfg_q.w_offset +
               ((cout_grp_q * n_cin_chks + cin_chk_q) *
                n_ksteps * n_ksteps + kh_q * n_ksteps + kw_q) *
               P_COUT_PAR * P_CIN_PAR;
    end
    w_addr_next = w_base + mac_cnt_q * P_CIN_PAR;
  end

  // ---------------------------------------------------------------------------
  // IFM read address generation
  // ---------------------------------------------------------------------------
  // Linear address: channel-major layout  [c][h][w]
  // in_h = out_h * stride - padding + kh
  // in_w = out_w * stride - padding + kw
  // For padded pixels: address = PADDING_ADDR_FLAG (returns 0 from buffer)
  logic [9:0]  in_h, in_w;
  logic        pad_pixel;

  always_comb begin
    automatic logic signed [10:0] tmp_h, tmp_w;
    tmp_h = (out_row_q * cfg_q.stride) - cfg_q.padding + kh_q;
    tmp_w = (out_col_q * cfg_q.stride) - cfg_q.padding + kw_q;

    if (tmp_h < 0 || tmp_h >= cfg_q.out_h * cfg_q.stride ||
        tmp_w < 0 || tmp_w >= cfg_q.out_w * cfg_q.stride) begin
      pad_pixel  = 1'b1;
      in_h       = '0;
      in_w       = '0;
    end else begin
      pad_pixel  = 1'b0;
      in_h       = tmp_h[9:0];
      in_w       = tmp_w[9:0];
    end

    // IFM address: [cin_chunk_base + cin_lane][in_h][in_w]
    // simplified: channel * (in_h_max * in_w_max) + row * in_w_max + col
    // in_h_max = out_h * stride (approximate; see NOTE below)
    // NOTE: precise in_h/in_w comes from cfg_q.in_shape stored in hw_graph;
    //       here we use out_shape × stride as an upper bound for addressing.
    automatic logic [9:0] in_h_max_w;
    in_h_max_w = cfg_q.out_w * cfg_q.stride;  // input width approximation
    ifm_rd_addr = (cin_chk_q * P_CIN_PAR + mac_cnt_q) *
                   (cfg_q.out_h * cfg_q.stride) * in_h_max_w +
                   in_h * in_h_max_w + in_w;
  end

  // ---------------------------------------------------------------------------
  // OFM write address generation
  // ---------------------------------------------------------------------------
  always_comb begin
    ofm_wr_addr = (cout_grp_q * P_COUT_PAR) *
                   cfg_q.out_h * cfg_q.out_w +
                   out_row_q * cfg_q.out_w + out_col_q;
  end

  // ---------------------------------------------------------------------------
  // Stage / node tracking for CSR
  // ---------------------------------------------------------------------------
  always_comb begin
    current_node = node_q;
    // Coarse stage mapping (matches spec §3.3.1)
    case (node_q)
      7'd0:              current_stage = 4'd1; // Stem
      7'd1  .. 7'd12:    current_stage = 4'd2; // BB-Stage1-2
      7'd13 .. 7'd28:    current_stage = 4'd3; // BB-Stage3-4 + SPPF
      7'd29 .. 7'd35:    current_stage = 4'd4; // PSA / model.10
      7'd36 .. 7'd57:    current_stage = 4'd6; // Neck PAN-FPN
      default:           current_stage = 4'd7; // Head
    endcase
  end

  // ---------------------------------------------------------------------------
  // Main FSM  (sequential)
  // ---------------------------------------------------------------------------
  always_ff @(posedge clock) begin
    if (reset) begin
      state_q    <= ST_IDLE;
      node_q     <= '0;
      out_row_q  <= '0;
      out_col_q  <= '0;
      cout_grp_q <= '0;
      cin_chk_q  <= '0;
      kh_q       <= '0;
      kw_q       <= '0;
      mac_cnt_q  <= '0;
      busy       <= 1'b0;
      frame_done <= 1'b0;
      mac_en     <= 1'b0;
      acc_clear  <= 1'b0;
      dw_mode    <= 1'b0;
      w_rd_en    <= 1'b0;
      b_rd_en    <= 1'b0;
      w_addr     <= '0;
      b_addr     <= '0;
      ofm_wr_en  <= 1'b0;
      skip_wr_en <= 1'b0;
      req_valid_in <= 1'b0;
    end else begin
      // Default deasserts
      frame_done   <= 1'b0;
      mac_en       <= 1'b0;
      acc_clear    <= 1'b0;
      w_rd_en      <= 1'b0;
      b_rd_en      <= 1'b0;
      ofm_wr_en    <= 1'b0;
      skip_wr_en   <= 1'b0;
      req_valid_in <= 1'b0;

      case (state_q)
        // ------------------------------------------------------------------
        ST_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            busy    <= 1'b1;
            node_q  <= '0;
            state_q <= ST_LOAD_CFG;
          end
        end

        // ------------------------------------------------------------------
        ST_LOAD_CFG: begin
          cfg_q      <= CFG_ROM[node_q];
          out_row_q  <= '0;
          out_col_q  <= '0;
          cout_grp_q <= '0;
          cin_chk_q  <= '0;
          kh_q       <= '0;
          kw_q       <= '0;
          mac_cnt_q  <= '0;
          dw_mode    <= CFG_ROM[node_q].depthwise;
          shift_bits <= CFG_ROM[node_q].shift_bits;
          act_sel    <= CFG_ROM[node_q].activation;
          bias_en    <= CFG_ROM[node_q].has_bias;

          case (CFG_ROM[node_q].op)
            OP_CONV:     state_q <= ST_PREFETCH_W;
            OP_MAXPOOL:  state_q <= ST_MAXPOOL;
            OP_UPSAMPLE: state_q <= ST_UPSAMPLE;
            default:     state_q <= ST_NEXT_NODE;
          endcase
        end

        // ------------------------------------------------------------------
        ST_PREFETCH_W: begin
          // Issue first weight ROM read (1-cycle latency)
          w_rd_en  <= 1'b1;
          w_addr   <= w_addr_next;
          state_q  <= ST_MAC_LOOP;
          if (cfg_q.has_bias) begin
            b_rd_en <= 1'b1;
            b_addr  <= cfg_q.b_offset + cout_grp_q * P_COUT_PAR;
          end
        end

        // ------------------------------------------------------------------
        // MAC inner loop:
        //   iterates over (out_row, out_col, cout_grp, cin_chk, kh, kw)
        //   innermost = mac_cnt (P_CIN_PAR reads per MAC cycle)
        // ------------------------------------------------------------------
        ST_MAC_LOOP: begin
          // Issue MAC
          mac_en    <= 1'b1;
          acc_clear <= (cin_chk_q == 0 && kh_q == 0 && kw_q == 0);
          w_rd_en   <= 1'b1;
          w_addr    <= w_addr_next;

          // Advance kernel position
          if (kw_q == cfg_q.kernel - 1) begin
            kw_q <= '0;
            if (kh_q == cfg_q.kernel - 1) begin
              kh_q <= '0;
              if (!cfg_q.depthwise && cin_chk_q == n_cin_chks - 1) begin
                cin_chk_q <= '0;
                // All accumulations for this output pixel+cout_grp done
                state_q   <= ST_BIAS_REQ;
              end else begin
                cin_chk_q <= cin_chk_q + 1;
              end
            end else begin
              kh_q <= kh_q + 1;
            end
          end else begin
            kw_q <= kw_q + 1;
          end
        end

        // ------------------------------------------------------------------
        ST_BIAS_REQ: begin
          // Trigger requant (bias fetched in ST_PREFETCH_W above)
          req_valid_in <= 1'b1;
          state_q      <= ST_WRITE_OFM;
        end

        // ------------------------------------------------------------------
        ST_WRITE_OFM: begin
          // Write P_COUT_PAR INT8 results to OFM buffer
          ofm_wr_en  <= 1'b1;
          ofm_wr_addr <= ofm_wr_addr;  // held from addr_gen

          // Advance spatial / cout_grp counters
          if (cout_grp_q == n_cout_grps - 1) begin
            cout_grp_q <= '0;
            if (out_col_q == cfg_q.out_w - 1) begin
              out_col_q <= '0;
              if (out_row_q == cfg_q.out_h - 1) begin
                // Layer complete → check for skip-write, then advance node
                out_row_q <= '0;
                state_q   <= ST_NEXT_NODE;
                // Save skip buffers at designated nodes
                case (node_q)
                  7'd12: begin skip_wr_en <= 1'b1; skip_sel <= 2'd0; end  // P3
                  7'd20: begin skip_wr_en <= 1'b1; skip_sel <= 2'd1; end  // P4
                  7'd35: begin skip_wr_en <= 1'b1; skip_sel <= 2'd2; end  // P5
                  7'd39: begin skip_wr_en <= 1'b1; skip_sel <= 2'd3; end  // P4_pan
                  default: ;
                endcase
              end else begin
                out_row_q <= out_row_q + 1;
                state_q   <= ST_PREFETCH_W;
              end
            end else begin
              out_col_q <= out_col_q + 1;
              state_q   <= ST_PREFETCH_W;
            end
          end else begin
            cout_grp_q <= cout_grp_q + 1;
            // Prefetch next bias group
            if (cfg_q.has_bias) begin
              b_rd_en <= 1'b1;
              b_addr  <= cfg_q.b_offset + (cout_grp_q + 1) * P_COUT_PAR;
            end
            state_q <= ST_PREFETCH_W;
          end
        end

        // ------------------------------------------------------------------
        ST_MAXPOOL: begin
          // 3×3 max-pool: implemented as a dedicated datapath in top-level.
          // The scheduler just signals mp_en; the max-pool unit reads from
          // the current active IFM bank and writes to OFM bank.
          // We stay here for out_h × out_w cycles then advance.
          if (out_row_q == cfg_q.out_h - 1 && out_col_q == cfg_q.out_w - 1) begin
            out_row_q <= '0;
            out_col_q <= '0;
            state_q   <= ST_NEXT_NODE;
          end else if (out_col_q == cfg_q.out_w - 1) begin
            out_col_q <= '0;
            out_row_q <= out_row_q + 1;
          end else begin
            out_col_q <= out_col_q + 1;
          end
        end

        // ------------------------------------------------------------------
        ST_UPSAMPLE: begin
          // Nearest-neighbour ×2 upsample: address remap only, no ALU.
          // Reads input at (r/2, c/2) for each (r, c) in the output.
          // The top-level FM buffer read address uses the upsample flag
          // to halve the IFM address bits.
          if (out_row_q == cfg_q.out_h - 1 && out_col_q == cfg_q.out_w - 1) begin
            out_row_q <= '0;
            out_col_q <= '0;
            state_q   <= ST_NEXT_NODE;
          end else if (out_col_q == cfg_q.out_w - 1) begin
            out_col_q <= '0;
            out_row_q <= out_row_q + 1;
          end else begin
            out_col_q <= out_col_q + 1;
          end
          // Copy IFM -> OFM with halved address
          ofm_wr_en   <= 1'b1;
          ofm_wr_addr <= out_row_q * cfg_q.out_w + out_col_q;
          ifm_rd_addr <= (out_row_q >> 1) * (cfg_q.out_w >> 1) + (out_col_q >> 1);
        end

        // ------------------------------------------------------------------
        ST_NEXT_NODE: begin
          if (node_q == N_NODES - 1) begin
            state_q <= ST_FRAME_DONE;
          end else begin
            node_q  <= node_q + 1;
            state_q <= ST_LOAD_CFG;
          end
        end

        // ------------------------------------------------------------------
        ST_FRAME_DONE: begin
          frame_done <= 1'b1;
          busy       <= 1'b0;
          state_q    <= ST_IDLE;
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

  // Bank-select: flip after each node completes
  // (simple toggle; the top-level buffers are double-banked)
  logic bank_toggle_q;
  always_ff @(posedge clock) begin
    if (reset)
      bank_toggle_q <= 1'b0;
    else if (state_q == ST_NEXT_NODE)
      bank_toggle_q <= ~bank_toggle_q;
  end

  assign ifm_bank_sel = bank_toggle_q;
  assign ofm_bank_sel = ~bank_toggle_q;

  // Skip buffer write address follows OFM write address
  assign skip_wr_addr = ofm_wr_addr;

endmodule : layer_scheduler
