// The former hand-written nppiColorTwist32f kernels (8u/16u/16s) lived here.
// They diverged from the measured NVIDIA NPP 13.4 semantics (C1 channel
// replication, unconditional +0.5f rounding, own validation rules) and were
// replaced by the shared executor in nppi_color_twist_executor.cu; the entry
// points in nppi_color_twist.cpp now delegate to it.  This file is kept as a
// placeholder so the source list stays stable.
