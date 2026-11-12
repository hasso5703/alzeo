# Qwen3.8 Flash-Next in Unsloth's GGUF on a DGX Spark (llama.cpp)

`unsloth/Qwen3.8-Flash-Next-GGUF`, Unsloth's GGUF of Qwen3.8-Flash-Next, in its dynamic 4-bit quantization UD-Q4_K_XL:
76.9 GiB of weights and the model's n-gram (PLE) table in IQ4_NL, 26.8 GiB, which stays on the disk and is read as the
tokens need it. This stack serves it with llama.cpp's own server (the project's arm64 CUDA image, b12316), its n-gram
lookup and ggml-org's GGUF of the DFlash2 draft, as a lane of its own beside the [flash lane](qwen38-flash-next.md) (SGLang) and the
[FP4-Local lane](qwen38-flash-next-fp4-local.md) (vLLM), which serve the same model.

## One command

```bash
./switch-model.sh flash-unsloth
```

## Measured on our box (ASUS Ascent GX10, 128 GB)

- **51.8 tok/s** in code, **56.9** in reasoning, **31.5** in prose and **123.7** in a rewrite of code, one request at the
  model card's sampling, with its n-gram lookup and its DFlash2 draft, on its pinned image (llama.cpp b12316, 2026-11-12,
  each text's first pass, the weights copied into the GPU's memory from a USB SSD); 27.9, 27.7, 27.3 and 27.4 without
  either, on the image measured before it (b12282).
- **63.2 tok/s** in all with two requests at once, each with a window of its own of 262,144 tokens, against 54.2 for one.
- The whole window read: three codes of three found in a prompt of 255,529 tokens; with YaRN, 524,288 tokens as an
  option, and `FLASH_UNSLOTH_CONTEXT=1m` given to its installer for **one request at a time with 1,048,576 tokens** (three codes of
  three in 985,296 tokens, the KV cache in q8_0).
- GSM8K **98.4 %** (250 questions) and HumanEval **99.4 %** (164 problems) at the medium effort and temperature 0, and
  20 tool calls right out of 20 (llama.cpp b12291 built on the box, the DFlash2 draft alone); MMLU not measured.
  TODO(measure): its quality in its 1M mode.
- **95.3 GiB** of the GPU's memory at ready with its two windows and its draft, **20 GiB** left to the box, 14.2 at the
  lowest under its probes; 98.5 GiB and 16.1 left in its 1M mode.
- A window shared by several requests (`-kvu`) is refused for now: llama-server stopped itself under load in four
  of the nine runs that shared one (2026-11-12, llama.cpp's open issue 31656; the lane page has the numbers).
- Ready in 35 to 50 s with its files on the internal disk, 50 to 281 s from a USB SSD.
- 117 GB on disk: 112 for its GGUF files and their draft, 4.5 for the image.
- Measures of minutes, run back to back: no continuous load was measured.

## What it is good at

The flash model as a GGUF on llama.cpp, in about half the flash lane's disk (117 GB against 223): a reference for
whoever runs the model that way, and the lane that follows Unsloth's and llama.cpp's releases. The two other flash
lanes are ahead on a long prompt (llama.cpp computes one about 2.6 times slower than the FP4-Local lane) and on several
requests at once.

## In the repo

- [docs/flash-unsloth-lane.md](../flash-unsloth-lane.md), the lane page: its files, the table on the disk, the draft
  measured, its template, what it does not have
- [BENCHMARKS.md](../../BENCHMARKS.md), every number with its method
