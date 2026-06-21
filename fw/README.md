# fw/ — host firmware / glue (placeholder)

Reserved for the host-side firmware and control glue that drives the Silicon YOLO
accelerator (DMA of the 640×640 input, kicking the streaming pipeline, reading
back detections). Populated during RTL bring-up; the model→HW handoff in
`hwgraph/`, `hwconst/`, and `golden/` does not depend on it.
