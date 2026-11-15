"""Test helpers. `tiny_qwen3vl` builds a small randomly initialised Qwen3-VL with the real architecture,
tokenizer and image processor, so the model-facing code (adapter, generation, KV capture) can be run
end to end on a CPU before it goes to the GPU server."""
