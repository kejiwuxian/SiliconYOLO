// =============================================================================
// yolov10n_pkg.sv  —  YOLOv10n Accelerator System Package
// =============================================================================
// Layer offsets derived from hw_graph.json via PowerShell offset computation.
// 83 Conv2D + 3 MaxPool (SPPF) + 2 Upsample = 88 compute nodes total.
// =============================================================================

package yolov10n_pkg;

  // ---------------------------------------------------------------------------
  // System parameters
  // ---------------------------------------------------------------------------
  localparam int unsigned CLK_FREQ_MHZ      = 200;
  localparam int unsigned P_MAC_WIDTH       = 1024;  // Total parallel MACs
  localparam int unsigned P_COUT_PAR        = 64;    // Output channels per cycle
  localparam int unsigned P_CIN_PAR         = 16;    // Input channels per cycle
  localparam int unsigned P_TILE_H          = 8;     // FM tile height
  localparam int unsigned P_TILE_W          = 8;     // FM tile width
  localparam int unsigned P_QUANT_SHIFT_MAX = 15;
  localparam int unsigned P_DETECTION_MAX   = 300;

  // Weight / bias ROM sizes (from hw_graph.json offset computation)
  localparam int unsigned WEIGHT_ROM_DEPTH  = 2290288; // INT8 weights total
  localparam int unsigned BIAS_ROM_DEPTH    =    8976; // INT32 biases total
  localparam int unsigned N_CONV_LAYERS     =      83;
  localparam int unsigned N_NODES           =      88; // Conv+MaxPool+Upsample

  // ---------------------------------------------------------------------------
  // Enumerated types
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    ACT_NONE = 2'd0,
    ACT_SILU = 2'd1,
    ACT_RELU = 2'd2
  } act_t;

  typedef enum logic [1:0] {
    OP_CONV     = 2'd0,
    OP_MAXPOOL  = 2'd1,
    OP_UPSAMPLE = 2'd2
  } op_t;

  // ---------------------------------------------------------------------------
  // Per-layer configuration record  (95 bits)
  // ---------------------------------------------------------------------------
  typedef struct packed {
    op_t         op;          // CONV / MAXPOOL / UPSAMPLE
    logic        depthwise;   // 1 = depthwise grouped conv
    act_t        activation;  // post-conv activation
    logic        int4_en;     // 1 = INT4 weight precision
    logic        has_bias;    // 1 = bias ROM entry present
    logic [9:0]  c_in;        // input  channels (1–512)
    logic [8:0]  c_out;       // output channels (1–256)
    logic [2:0]  kernel;      // kernel size (1, 3, or 7)
    logic [1:0]  stride;      // stride (1 or 2)
    logic [1:0]  padding;     // zero-padding each side (0, 1, or 3)
    logic [9:0]  out_h;       // output feature-map height
    logic [9:0]  out_w;       // output feature-map width
    logic [3:0]  shift_bits;  // INT32->INT8 right-shift (0–15)
    logic [21:0] w_offset;    // byte offset in unified weight ROM
    logic [13:0] b_offset;    // index offset in INT32 bias ROM
  } layer_cfg_t;

  // ---------------------------------------------------------------------------
  // Config ROM initialiser function — all 88 nodes
  // ---------------------------------------------------------------------------
  // Struct literal field order matches layer_cfg_t packed definition:
  //  {op, dw, act, int4, has_b, c_in, c_out, k, s, pad, oh, ow, sh, woff, boff}
  // ---------------------------------------------------------------------------
  function automatic layer_cfg_t [0:N_NODES-1] init_cfg_rom();
    layer_cfg_t [0:N_NODES-1] r;
    // ---- Backbone layers 0-12 ------------------------------------------------
    r[ 0]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd3,  9'd16, 3'd3,2'd2,2'd1,10'd320,10'd320,4'd10,22'd0,      14'd0   };
    r[ 1]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd16, 9'd32, 3'd3,2'd2,2'd1,10'd160,10'd160,4'd9, 22'd432,    14'd16  };
    r[ 2]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd32, 9'd32, 3'd1,2'd1,2'd0,10'd160,10'd160,4'd8, 22'd5040,   14'd48  };
    r[ 3]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd16, 9'd16, 3'd3,2'd1,2'd1,10'd160,10'd160,4'd5, 22'd6064,   14'd80  };
    r[ 4]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd16, 9'd16, 3'd3,2'd1,2'd1,10'd160,10'd160,4'd8, 22'd8368,   14'd96  };
    r[ 5]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd48, 9'd32, 3'd1,2'd1,2'd0,10'd160,10'd160,4'd7, 22'd10672,  14'd112 };
    r[ 6]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd64, 3'd3,2'd2,2'd1,10'd80, 10'd80, 4'd9, 22'd12208,  14'd144 };
    r[ 7]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd64, 9'd64, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd7, 22'd30640,  14'd208 };
    r[ 8]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd32, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd7, 22'd34736,  14'd272 };
    r[ 9]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd32, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd9, 22'd43952,  14'd304 };
    r[10]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd32, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd8, 22'd53168,  14'd336 };
    r[11]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd32, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd8, 22'd62384,  14'd368 };
    r[12]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd128,9'd64, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd8, 22'd71600,  14'd400 };
    // ---- Backbone layers 13-28 (include DW strides + SPPF) ------------------
    r[13]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd64, 9'd128,3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd8, 22'd79792,  14'd464 };
    r[14]='{OP_CONV,1'b1,ACT_NONE,1'b1,1'b1,10'd128,9'd128,3'd3,2'd2,2'd1,10'd40, 10'd40, 4'd7, 22'd87984,  14'd592 };
    r[15]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd128,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd89136,  14'd720 };
    r[16]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd7, 22'd105520, 14'd848 };
    r[17]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd9, 22'd142384, 14'd912 };
    r[18]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd9, 22'd179248, 14'd976 };
    r[19]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd8, 22'd216112, 14'd1040};
    r[20]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd256,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd9, 22'd252976, 14'd1104};
    r[21]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd128,9'd256,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd285744, 14'd1232};
    r[22]='{OP_CONV,1'b1,ACT_NONE,1'b1,1'b1,10'd256,9'd256,3'd3,2'd2,2'd1,10'd20, 10'd20, 4'd8, 22'd318512, 14'd1488};
    r[23]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd256,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd320816, 14'd1744};
    r[24]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd128,9'd128,3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd9, 22'd386352, 14'd2000};
    r[25]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd128,9'd128,3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd9, 22'd533808, 14'd2128};
    r[26]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd384,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd681264, 14'd2256};
    r[27]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd256,9'd128,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd779568, 14'd2512};
    r[28]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd512,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd10,22'd812336, 14'd2640};
    // ---- PSA Transformer block 29-35 ----------------------------------------
    r[29]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd256,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd943408, 14'd2896};
    r[30]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd128,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd1008944,14'd3152};
    r[31]='{OP_CONV,1'b1,ACT_NONE,1'b1,1'b1,10'd128,9'd128,3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd7, 22'd1041712,14'd3408};
    r[32]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd128,9'd128,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd1042864,14'd3536};
    r[33]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd128,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd9, 22'd1059248,14'd3664};
    r[34]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd256,9'd128,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd1092016,14'd3920};
    r[35]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd256,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd10,22'd1124784,14'd4048};
    // ---- Neck PAN-FPN 36-57 -------------------------------------------------
    r[36]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd384,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd1190320,14'd4304};
    r[37]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd8, 22'd1239472,14'd4432};
    r[38]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd8, 22'd1276336,14'd4496};
    r[39]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd192,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd1313200,14'd4560};
    r[40]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd192,9'd64, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd8, 22'd1337776,14'd4688};
    r[41]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd32, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd8, 22'd1350064,14'd4752};
    r[42]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd32, 9'd32, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd8, 22'd1359280,14'd4784};
    r[43]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd96, 9'd64, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd8, 22'd1368496,14'd4816};
    r[44]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd2,2'd1,10'd40, 10'd40, 4'd8, 22'd1374640,14'd4880};
    r[45]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd192,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd1411504,14'd4944};
    r[46]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd8, 22'd1436080,14'd5072};
    r[47]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd8, 22'd1472944,14'd5136};
    r[48]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd192,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd1509808,14'd5200};
    r[49]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd128,9'd128,3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd7, 22'd1534384,14'd5328};
    r[50]='{OP_CONV,1'b1,ACT_NONE,1'b1,1'b1,10'd128,9'd128,3'd3,2'd2,2'd1,10'd20, 10'd20, 4'd8, 22'd1550768,14'd5456};
    r[51]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd384,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd7, 22'd1551920,14'd5584};
    r[52]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd128,9'd128,3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd7, 22'd1650224,14'd5840};
    r[53]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd128,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd7, 22'd1651376,14'd5968};
    r[54]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd256,9'd256,3'd7,2'd1,2'd3,10'd20, 10'd20, 4'd8, 22'd1684144,14'd6224};
    r[55]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd256,9'd128,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd1696688,14'd6480};
    r[56]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd128,9'd128,3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd7, 22'd1729456,14'd6608};
    r[57]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd384,9'd256,3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd1730608,14'd6736};
    // ---- Detection head cv2 (regression) 58-66 ------------------------------
    r[58]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd8, 22'd1828912,14'd6992};
    r[59]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd8, 22'd1865776,14'd7056};
    r[60]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd64, 9'd64, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd8, 22'd1902640,14'd7120};
    r[61]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd128,9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd9, 22'd1906736,14'd7184};
    r[62]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd7, 22'd1980464,14'd7248};
    r[63]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd64, 9'd64, 3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd9, 22'd2017328,14'd7312};
    r[64]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd256,9'd64, 3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd9, 22'd2021424,14'd7376};
    r[65]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd9, 22'd2168880,14'd7440};
    r[66]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd64, 9'd64, 3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd7, 22'd2205744,14'd7504};
    // ---- Detection head cv3 (classification) 67-81 -------------------------
    r[67]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd64, 9'd64, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd7, 22'd2209840,14'd7568};
    r[68]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd64, 9'd80, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd8, 22'd2210416,14'd7632};
    r[69]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd80, 9'd80, 3'd3,2'd1,2'd1,10'd80, 10'd80, 4'd7, 22'd2215536,14'd7712};
    r[70]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd80, 9'd80, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd7, 22'd2216256,14'd7792};
    r[71]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd80, 9'd80, 3'd1,2'd1,2'd0,10'd80, 10'd80, 4'd9, 22'd2222656,14'd7872};
    r[72]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd128,9'd128,3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd7, 22'd2229056,14'd7952};
    r[73]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd128,9'd80, 3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd8, 22'd2230208,14'd8080};
    r[74]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd80, 9'd80, 3'd3,2'd1,2'd1,10'd40, 10'd40, 4'd8, 22'd2240448,14'd8160};
    r[75]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd80, 9'd80, 3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd7, 22'd2241168,14'd8240};
    r[76]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd80, 9'd80, 3'd1,2'd1,2'd0,10'd40, 10'd40, 4'd10,22'd2247568,14'd8320};
    r[77]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd256,9'd256,3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd5, 22'd2253968,14'd8400};
    r[78]='{OP_CONV,1'b0,ACT_SILU,1'b0,1'b1,10'd256,9'd80, 3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd8, 22'd2256272,14'd8656};
    r[79]='{OP_CONV,1'b1,ACT_SILU,1'b1,1'b1,10'd80, 9'd80, 3'd3,2'd1,2'd1,10'd20, 10'd20, 4'd9, 22'd2276752,14'd8736};
    r[80]='{OP_CONV,1'b0,ACT_SILU,1'b1,1'b1,10'd80, 9'd80, 3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd6, 22'd2277472,14'd8816};
    r[81]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b1,10'd80, 9'd80, 3'd1,2'd1,2'd0,10'd20, 10'd20, 4'd10,22'd2283872,14'd8896};
    // ---- DFL conv 82 --------------------------------------------------------
    r[82]='{OP_CONV,1'b0,ACT_NONE,1'b1,1'b0,10'd16, 9'd1,  3'd1,2'd1,2'd0,10'd4,  10'd8400,4'd1,22'd2290272,14'd0   };
    // ---- SPPF MaxPool 83-85 (kernel=5, pad=2, stride=1) --------------------
    r[83]='{OP_MAXPOOL,1'b0,ACT_NONE,1'b0,1'b0,10'd128,9'd128,3'd3,2'd1,2'd2,10'd20,10'd20,4'd0,22'd0,14'd0};
    r[84]='{OP_MAXPOOL,1'b0,ACT_NONE,1'b0,1'b0,10'd128,9'd128,3'd3,2'd1,2'd2,10'd20,10'd20,4'd0,22'd0,14'd0};
    r[85]='{OP_MAXPOOL,1'b0,ACT_NONE,1'b0,1'b0,10'd128,9'd128,3'd3,2'd1,2'd2,10'd20,10'd20,4'd0,22'd0,14'd0};
    // ---- Upsample 86-87 (nearest-neighbour x2) ------------------------------
    r[86]='{OP_UPSAMPLE,1'b0,ACT_NONE,1'b0,1'b0,10'd256,9'd256,3'd1,2'd1,2'd0,10'd40,10'd40,4'd0,22'd0,14'd0};
    r[87]='{OP_UPSAMPLE,1'b0,ACT_NONE,1'b0,1'b0,10'd128,9'd128,3'd1,2'd1,2'd0,10'd80,10'd80,4'd0,22'd0,14'd0};
    return r;
  endfunction

  // Package-level constant ROM
  localparam layer_cfg_t [0:N_NODES-1] CFG_ROM = init_cfg_rom();

endpackage : yolov10n_pkg
