# The Qwen3.8 Flash-Next Unsloth GGUF lane: the flash model on llama.cpp

`unsloth/Qwen3.8-Flash-Next-GGUF` is Unsloth's GGUF of Qwen3.8-Flash-Next, every quantization of the model in one
repository. This lane serves its dynamic 4-bit one, UD-Q4_K_XL, with llama.cpp's own server, `llama-server`, from the
image the llama.cpp project publishes for arm64 with CUDA, pinned by digest, with llama.cpp's n-gram lookup and then
ggml-org's GGUF of the DFlash2 draft of the model (Q8_0) proposing tokens. It serves on the text lanes' port, behind the same proxy and key, as model
`qwen3.8-flash-next-unsloth`, OpenAI or Anthropic dialect alike, and, as on the other Qwen lanes, `lean` is the reasoning
effort a request that names none gets. It is the first lane of this repository on llama.cpp, and a lane of its own beside
the two other lanes of the same model: the flash lane (SGLang, RadixArk's NVFP4 export, `docs/flash-lane.md`) and the
Qwen3.8 Flash-Next FP4-Local lane (vLLM, NVIDIA's FP4 build, `docs/flash-local-lane.md`).

Every number on this page was measured on the reference box (an ASUS Ascent GX10: GB10, 124 GiB of unified memory, its
Hugging Face cache on an external USB SSD, its home directory on the internal NVMe) on 2026-11-11 and 12, with llama.cpp
built on the box (v0.9.0, commit cd959ba, then b12291, commit d2b3eb9) unless the line names one of the project's
images: b12316's, the pinned one, or b12282's, measured before it. They are
measures of minutes each, run back to back through the night of 11 to 12/11, during which the box's hottest zone peaked
at 88 to 95.5 °C in ten of the twelve sets, the GPU drawing up to 72.4 W (the box's own telemetry): no continuous
load was measured, and none is promised here.

## The files

The repository holds every quantization of the model, hundreds of GB; the lane takes four files of one folder and the
draft, each pinned by its sha256 and its size, read from the Hub's LFS pointers at the pinned revisions and from the
reference box's own cache (`install-flash-unsloth.sh`; `check-pins.sh` asks the pointers every day):

| file | what it holds | size |
|---|---|---|
| `UD-Q4_K_XL/...-00001-of-00004.gguf` | the header: the model's metadata and its chat template | 10.9 MB |
| `UD-Q4_K_XL/...-00002-of-00004.gguf` and `-00003-` | the weights, 76.9 GiB | 49.7 GB and 32.8 GB |
| `UD-Q4_K_XL/...-00004-of-00004.gguf` | the n-gram (PLE) table, in IQ4_NL | 28.8 GB (26.8 GiB) |
| `dflash-Qwen3.8-Flash-Next-Q8_0.gguf` (`ggml-org/Qwen3.8-Flash-Next-GGUF`) | the DFlash2 draft | 0.65 GB |

They are fetched by `gguf-fetch.py`, with Python's standard library on the host: the server image has no Python (its
`server` stage is a CUDA runtime, curl and `llama-server`), so `ckpt-fetch.py`, every other lane's fetch, cannot run in it,
and `huggingface_hub` is not on the host. `gguf-fetch.py` writes the cache's own layout (`blobs/<sha256>` and a relative
link in `snapshots/<revision>/`), so a file `huggingface_hub` fetched before is the same file, with not a byte downloaded
again; it resumes a cut download, checks every file against its sha256 before it takes its name, refuses before the
first byte when the disk lacks the room, takes `huggingface_hub`'s own lock of each blob, and sends `HF_TOKEN`, read from
its environment, to the Hub only, never past its redirect to the CDN.

## The n-gram table stays on the disk

Without options, llama.cpp loads this model with its 26.8 GiB n-gram table in memory: on the GB10, an integrated GPU,
its `auto` load mode turns off both mmap and its lazy reading (`ggml-cuda.cu` types an integrated GPU as such, and
`llama-model.cpp` then loads without mmap), and at 131,072 tokens of context the box's memory guard stopped the load with
5 GiB free. With `--lazy-mode on`, the table is read from the disk row by row as the tokens need it, "for example
per-layer embeddings" as the option's own help says, whatever the load mode (b12291 maps a lazy tensor in every mode,
`init_mappings` in its `llama-model-loader.cpp`, and so does the pinned b12316, 25 commits after it, none of them in
the loader):

| 2026-11-11, v0.9.0, `-c 131072 -np 1` | ready | MemAvailable at ready | GPU | one stream (code, reasoning, prose) |
|---|---|---|---|---|
| no option | never: stopped by the guard at 5 GiB free | | | |
| `--load-mode mmap --lazy-mode on` | 101 s | 35 GiB | 82.5 GiB | 27.1 / 27.1 / 27.1 tok/s |
| the same and `--no-host` | 101 s | 35 GiB | 82.5 GiB | 27.3 / 27.2 / 26.9 tok/s |

The process itself holds almost nothing (0.3 GiB resident at ready). `--no-host` adds nothing, and the lane does not take
it; it takes `--no-repack`, which only concerns the tensors left on the processor's side, the table: repacking them
would copy what is read in place.

How the weights are read decides their speed. With `--load-mode mmap` the GPU reads them in place, in the page cache,
and from the same files it decoded at 27.9 tok/s in the evening's runs and at about 25 in the night's; with `none` they
are copied into its memory. Side by side on 2026-11-12, the weights on the internal disk, no draft, temperature 0, the
mean of four probes:

| `--load-mode`, with `--lazy-mode on` | decode | ready | MemAvailable at ready |
|---|---|---|---|
| `mmap` | 24.9 tok/s | 273 s | 32 GiB |
| `auto` | 27.3 tok/s | 40 s | 30 GiB |
| **`none`, the lane's** | **27.6 tok/s** | **35 s** | **31 GiB** |
| `dio` | 27.5 tok/s | 25 s | 30 GiB |

The three after `mmap` found its weights in the page cache (`dio` reads past it): their times are not those of a cold
start. Where the files sit made no difference to the decode: in `mmap`, the weights read from the USB SSD decoded at
24.8 tok/s in code and copied to the internal disk at 25.0, and with the table read through into the page cache before
the start, 24.7, or once ready, 24.5 (11.9 of its 26.8 GiB stayed there, 0.2 after the probes). Reading the weights
through once while the server starts made two loads 10 to 19 % shorter, which is no sure gain. So the lane reads its
files where the cache keeps them, with no copy and no preload.

A start from the USB SSD can be slow. In `none`, the images of b12316 and b12282 were ready in 261 to 281 s in three
starts on 2026-11-12, and in 50 and 90 s in two others; from the internal disk, 35 to 50 s. After each of the three
slow starts the first answers came slower too: in code, 23.2 tok/s at the first pass against 28.2 at the second with no
draft, 36.1 against 58.5 and 34.2 against 40.4 with the DFlash2 draft, and the reasoning that came next 48.1 against
62.6 and 41.6 against 56.3, which fits the table being read from the disk as the tokens need it. How much of a slow
start the page cache explains and how much the disk was not measured. Several requests at once with the table on the
USB SSD (the study's measures of them read it from the internal disk): GSM8K's 250 questions, eight requests at once
through the proxy on the lane's two windows, generated 62.9 tok/s in all (62.8 the day before), where one request alone
decodes at 64.1 in `bench.sh`'s greedy probes: two requests at once gave no more in all than one, as eight requests of
the Qwen3.8 Flash-Next FP4-Local lane decoded no faster than one from a USB SSD (which is why its Load copies its table
beside the home directory when the cache is on another disk). The texts differ (GSM8K reasons at the `medium` effort),
and how much of it the disk explains was not measured.

## The engine: llama.cpp's own image

`ghcr.io/ggml-org/llama.cpp@sha256:a5600a59...` is the arm64 image of the project's tag `server-cuda13-b12316`
(build b12316, commit 5d4c8eeb2, created on 2026-11-12), its labels and the tag's index read on the reference box. It is
built on CUDA 13.5 and asks for CUDA 13.5 or a driver of the 535, 580, 595, 610 or 615 branch (its
`NVIDIA_REQUIRE_CUDA`): the box's 580.178.04 runs it. Nothing is built on the box (the speech lane builds its own
llama.cpp for the CPU instead, `install-speech.sh`). It replaced the image of b12282, measured first, the same morning: on
b12282's image the n-gram lookup slowed the DFlash2 draft down, on b12316's it does not (the draft, below).

b12282's image decoded as b12291 built on the box, at temperature 0, the weights on the USB SSD for both (the image on
2026-11-12 from 00:20 to 00:37, b12291 on 2026-11-11 from 22:29 to 23:05):

| | code | reasoning | prose | rewrite |
|---|---|---|---|---|
| b12282's image, no draft | 27.9 | 27.7 | 27.9 | 27.4 tok/s |
| b12291 built on the box, no draft | 27.9 | 27.8 | 27.8 | 27.3 tok/s |
| b12282's image, DFlash2 Q8 | 58.6 | 56.3 | 25.4 | 49.6 tok/s |
| b12291 built on the box, DFlash2 Q8 | 60.3 | 69.0 | 27.0 | 52.0 tok/s |

Their answers are sound and part from each other's at a token where two were nearly tied (another compiler, another
CUDA), and so does their draft's acceptance in the reasoning probe (58 % against 73 %). Both ran in `mmap` and later
that night both decoded slower, still alike (the image 25.3 tok/s in code from 03:05, b12291 24.8 and 25.0 from 01:48 to
02:18): the load mode, above.

The unit runs it as every lane runs its engine (`qwen38-flash-unsloth.service.template` says why for each option):

- `-ngl 999 -fa on --no-repack`: every layer on the GPU, flash attention.
- `--jinja --chat-template-file`: the GGUF's chat template with the fixes below, mounted read-only.
- `--load-mode none --lazy-mode on`: the weights copied into the GPU's memory, the table read from the disk, above.
- `-c 524288 -np 2`: two requests at once, each with the window the model was trained for (`qwen4exp.context_length`
  in the GGUF), 262,144 tokens, its KV cache in llama.cpp's own f16, below. `FLASH_UNSLOTH_CONTEXT=1m` renders `-c
  1048576 -np 1 -ctk q8_0 -ctv q8_0 --rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 4` instead: one request at
  a time, with 1,048,576 tokens, below.
- `-md`, `--spec-type ngram-mod,draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0.5 --spec-draft-sampling
  probabilistic`: the n-gram lookup and the draft, below.
- Each of these is one variable of the installer (`FLASH_UNSLOTH_LOAD`, `_SLOTS`, `_SHARED`, `_CTX`, `_YARN`, `_KV`,
  `_SPEC`), with its measure beside it, and its rendering refuses a value the engine would not take, and a window the
  requests share (below).
- `--alias qwen3.8-flash-next-unsloth`: the name it serves under (`llama-server` has no `--served-model-name`).
- `--metrics --no-ui --api-key-file /out/api-key --host 127.0.0.1 --port 30000`: its Prometheus counters on, its web
  interface off, the box's key from the file mounted read-only (in no command line), asked on every route but `/health`
  and `/v1/health` (read in its `server-http.cpp`), on the text port of this box alone. `LLAMA_ARG_HOST` and
  `LLAMA_ARG_PORT` are set to the same values, since the image sets 0.0.0.0 and 8080; and `--no-healthcheck`, since the
  image's own check asks port 8080.
- The Hugging Face cache read-only, and `--memory 110g`, the cap of the other lanes. The study ran without it; under it,
  in the lane's proofs (2026-11-12 and 13), the lane was ready 52 to 122 s after another lane stopped and 52 to 55 s on
  a restart, and `bench.sh` gave a greedy median of 64.1 tok/s on a freshly started engine (code 63.3,
  reasoning 64.9, math at temperature 0.6 58.7, prose 30.9 apart from the median, each probe's first run).

The model card's sampling (temperature 1.0, top_p 0.95, top_k 20) is in the GGUF, and the server applies it to a request
that sets none (read in `/props`).

## The draft: n-grams, then DFlash2 Q8, kept from 0.5, checked by rejection

Measured one stream at a time on the files of the lane, 400 tokens (600 for the rewrite), the median of three passes.
At temperature 0 (2026-11-11, v0.9.0), no draft won everywhere: DFlash2 Q8 led in reasoning (68.0 tok/s) and code, the
MTP head in the rewrite (57.7, 99 % accepted), their means within the noise (48.6 to 50.7 against 27.2 without a draft).
Unsloth's "shared" MTP head does not load (it carries no token embeddings), and DFlash2 in Q4_0 accepts less than in Q8_0.
At the model card's sampling, what a client that sets nothing gets (2026-11-11, b12291, seeds 7, 8 and 9):

| | code | reasoning | prose | mean |
|---|---|---|---|---|
| no draft | 24.6 | 24.5 | 24.2 | 24.4 tok/s |
| DFlash2 Q8, no threshold, exact match | 54.1 (60 %) | 51.8 (58 %) | 22.8 (16 %) | 42.9 |
| DFlash2 Q8, no threshold, rejection sampling | 51.1 (54 %) | 57.4 (65 %) | 25.0 (20 %) | 44.5 |
| DFlash2 Q8, threshold 0.5, exact match | 54.1 (66 %) | 55.9 (70 %) | 27.2 (36 %) | 45.7 |
| **DFlash2 Q8, threshold 0.5, rejection sampling** | **56.8 (69 %)** | **54.6 (69 %)** | **29.7 (41 %)** | **47.0** |
| DFlash2 Q8, threshold 0.75, rejection sampling | 51.3 (82 %) | 57.5 (83 %) | 28.9 (66 %) | 45.9 |
| MTP Q8 n=4, rejection sampling | 48.6 (78 %) | 50.1 (82 %) | 29.5 (39 %) | 42.7 |

(in brackets, the share of the draft's tokens accepted). The lane takes the bold line: 1.9 times the speed without a
draft at the card's sampling, 2.3 times in code. The draft proposes 7 tokens at most, its block of 8 less the anchor (the
server brings a larger value back to 7). On b12282's image, in the lane's load mode, where the weights decode faster
without a draft, it gave 1.7 times that speed, 1.9 times in code (the second table below).

Before the draft, llama.cpp looks the next tokens up in the n-grams of the text it has processed (`--spec-type
ngram-mod,draft-dflash`: a pool of about 16 MB of n-gram hashes, each with the token that followed, shared by every
request, as its `docs/speculative.md` says): a text the model writes again, as when it rewrites code it was given, comes
out more than twice as fast, and the rest within a few percent. Every token drafted is checked by the model as the
draft's are, so the answers are the model's; what the pool shares between requests is speed, as a prefix cache does. At
the model card's sampling, one stream, the first pass of each text, since the pool keeps what a pass wrote (2026-11-12,
b12291 built on the box, in `mmap`, `choisi-ngram` against `choisi`):

| | code | reasoning | prose | rewrite of code |
|---|---|---|---|---|
| DFlash2 Q8 alone | 56.1 (69 %) | 54.7 (69 %) | 29.4 (41 %) | 48.2 (66 %) tok/s |
| n-grams, then DFlash2 Q8 | 62.4 (55 %) | 52.0 (51 %) | 29.6 (40 %) | 121.0 (80 %) tok/s |

A second run the same night gave the same (120.7 against 48.6 in the rewrite); under v0.9.0 the same combination cost
every turn. On the images, the same way, in the lane's load mode (`none`) with the weights on the USB SSD (2026-11-12,
from 07:27 and from 08:40; the median of three passes, each text's first pass for the lookup):

| | code | reasoning | prose | rewrite of code |
|---|---|---|---|---|
| b12282's image, no draft | 27.9 | 27.7 | 27.3 | 27.4 tok/s |
| b12282's image, DFlash2 Q8 alone | 53.8 (65 %) | 58.8 (70 %) | 31.3 (41 %) | 52.0 (66 %) tok/s |
| b12282's image, n-grams, then DFlash2 Q8 | 46.9 (58 %) | 43.9 (66 %) | 22.9 (43 %) | 116.2 (78 %) tok/s |
| **b12316's image, the lane's: n-grams, then DFlash2 Q8** | **51.8 (63 %)** | **56.9 (66 %)** | **31.5 (43 %)** | **123.7 (79 %) tok/s** |

On b12282's image the lookup slowed the draft down everywhere but in the rewrite; on b12316's it does not, with the same
tokens drafted and accepted in reasoning and in prose (319 of 484 and 209 of 488 on both images): the lane pins b12316
and takes the pair (`FLASH_UNSLOTH_SPEC` in the installer), within 4 % of the draft alone on b12282's image in code and
in reasoning, as fast in prose, 2.4 times as fast in a rewrite of code. The draft alone on b12316's image is no witness:
that start was one of the slow ones (261 s), and so were its first answers (34.2 tok/s in code, then 40.4 and 42.3,
where b12282's image gave 52.5 to 56.6). The draft alone on b12316's image after a quick start was not measured.

## Quality

Measured on 2026-11-12 with llama.cpp b12291 built on the box, the lane's files and its DFlash2 draft without the
n-gram lookup, eight requests at once of 20,480 tokens each, at the `medium` effort and temperature 0: GSM8K 98.4 %
(250 questions, 299 tokens an answer), HumanEval 99.4 % (164 problems, pass@1, 621 tokens an answer), and 20 tool calls
right out of 20. MMLU was not measured. These ran without YaRN and with the KV cache in f16, as the lane serves by
default. In its 1M mode (YaRN at four times, the KV cache in q8_0), through the proxy on the pinned image (the lane's
second proof, 2026-11-13, eight requests at once on its one window): GSM8K 98.0 % (250 questions) and HumanEval 98.2 % (164 problems, pass@1), at the same effort and
temperature; at its native window the same proof gave GSM8K 98.8 % again. Those runs and the native GSM8K were 75 min of
continuous load: the box's hottest zone peaked at 94.4 °C, at 90 °C or more in 109 of 2,256 readings, no throttling
flagged; the GPU clock cap of the cockpit's Machine view lowers it (2,000 MHz: 8 °C less on the FP4-Local lane's load,
[cockpit](cockpit.md)).

## The chat template

The GGUF carries Qwen's template with Unsloth's changes: it merges the system and developer messages that open a
conversation into one system turn, maps the effort `high` to `xhigh`, defaults to `xhigh`, and still raises "System
message must be at the beginning." for a system message past those, which opencode and Claude Code send. The lane
serves it with the fixes the other Qwen lanes' templates have (`patch-template.py --gguf`, which reads the template from
the GGUF's first part, its header alone): a later system message becomes a `<system-reminder>` block, a Claude Code
ToolSearch result (`tool_reference` blocks) a marker instead of a 500, the effort levels `max` and `minimal` mapped, and
the `lean` level, its default (`LEAN.md`). Its four levels are offered to opencode and pi as on the 27B: `lean`, `low`,
`medium`, `xhigh`.

## Several requests at once

Two requests at once, each with a window of its own of 262,144 tokens (`-c 524288 -np 2`, `FLASH_UNSLOTH_SLOTS=2` and
`FLASH_UNSLOTH_SHARED=0` in the installer; Hasan's decision 1318, 2026-11-12): llama-server decodes the tokens of both
in one batch at each step (its `update_slots`), a third request waits until one of them ends (the cockpit shows the
requests waiting, read off its `/metrics`), and no request can make the other fail. A request of 262,144 tokens or
more is refused at once ("exceeds the available context size"). Measured on 2026-11-12 before 12:23 (b12291 built on
the box, the weights on the internal disk, the DFlash2 draft alone, the KV cache in f16 unless the line says q8_0):

| | one request | two at once, in all (each) | four at once, in all (each) | eight at once, in all (each) | MemAvailable at ready, after the probes |
|---|---|---|---|---|---|
| **`-c 524288 -np 2`, the lane's** | **54.2 tok/s** | **63.2 (34.2)** | **66.9 (36.8), two of them waiting** | | **20 GiB, 14** |
| `-c 1048576 -np 4`, the KV cache in q8_0 | 53.0 tok/s | | 68.9 (18.2) | | 14 GiB, 10 |
| `-c 262144 -np 1` | 53.4 tok/s | | 56.2 (queued, one at a time) | | 28 GiB, 22 |
| `-c 262144 -np 4 -kvu`, one window shared: refused, below | 53.9 tok/s | | 72.9 (19.7) | 66.8 (18.7) | 25 GiB, 18 |
| `-c 262144 -np 8 -kvu`, the same | 54.5 tok/s | | 72.6 (19.1) | 69.2 (9.2) | 21 GiB, 14 |

With the lane's two windows, an agent's conversation of five turns came back from the cache with 61 to 77 tokens
computed again per turn after the first, two conversations taking turns with 19 and 20, each keeping its window's
cache, and the box kept 14.2 GiB at the lowest (its telemetry). Four windows in q8_0
left 10 GiB available after the probes: too thin a margin for this box. On the pinned image b12316 the two windows in
q8_0 took four salvoes of 4 and 8 requests and none stopped (63.2 to 65.3 tok/s in all, under the GPU clock cap of
2,200 MHz set from the cockpit at 12:23 that day), two conversations taking turns came back with 19 and 20 tokens
computed again, and the box kept 24 GiB at ready and 16.1 at the lowest (variantes-20261112-1501-kvu-image-image-b12316).

### One window shared by several requests: refused for now

It was meant to serve more requests at once on more context: up to four requests sharing one window (`-kvu`) of
524,288 or 1,048,576 tokens by YaRN (Hasan's decisions 1313 to 1317, 2026-11-12). Of the nine runs where requests
shared a window that day, llama-server stopped itself under load in four, with b12291 built on the box and with the
pinned image b12316: "common/common.cpp:1634: failed to remove sequence N with p0=..., p1=-1", an abort in
`common_memory::seq_rm` called from `post_decode`, after 4 minutes of GSM8K at 8 requests, at a salvo of 8 requests,
after 106 s of GSM8K, and on the image after a first salvo of 8 had passed. The five others, salvos of 1, 4 and 8 short requests of 3 to 5
minutes each, ended well: the abort is intermittent, and a form that may stop under its users' load is not served. It
is llama.cpp's open issue
[31656](https://github.com/ggml-org/llama.cpp/issues/31656): a draft's rejected tokens are rolled back from this hybrid
model's per-token recurrent snapshots (`llama_memory_recurrent::seq_rm`), which a sequence loses when `find_slot` moves
its cell between the parts of a batch, and a shared window splits a batch out of the sequences' order
(`llama_memory_hybrid_idx::init_batch`); whatever became of idle conversations, it stopped. What was measured before
it stopped, all under the GPU clock cap of 2,200 MHz (the memory does not depend on it):

| one window shared by four requests | available at ready, at the lowest | one request | four at once, in all | how it ended |
|---|---|---|---|---|
| 524,288 by YaRN at twice, the KV cache in q8_0, `--cache-ram 4096` | 22.2 GiB, 15.8 | 53.2 tok/s | 72.6 | stopped at a salvo of 8 requests, and after 106 s of GSM8K |
| the same, idle conversations kept in the window (`--no-cache-idle-slots --cache-ram 1024`) | 20.1 GiB, 16.9 | 47.2 tok/s | 68.1 | stopped after 4 minutes of GSM8K |
| the same as the first, on the image b12316 | 22.2 GiB, 16.8 | | 49.2 at eight | stopped after that salvo |
| 1,048,576 by YaRN at four times, the KV cache in q8_0 | 5 to 11 GiB, then under 6 | | | the study's memory guard |
| the same, `--cache-ram 1024` | 11.2 GiB, 8.3 | 38.1 tok/s | 59.5 | short probes only |

The installer refuses its setting `FLASH_UNSLOTH_SHARED` at 1 (a line of the installer, as its other settings are:
the environment does not set it) and says why. The form comes back when llama.cpp fixes the rollback
(its pull request 31669 reworks it), measured again then.

## Memory

At ready, with its two windows, the lane holds 97,552 MiB of the GPU's (95.3 GiB) and the box kept 20 GiB available,
14.2 at the lowest under the probes of the table above (b12291, 2026-11-12, a quiet box); on the box as it ships, with
the rest of its services running, 12.8 to 19.6 GiB at ready in the lane's proofs (2026-11-12 and 13); with one window, 88,693 MiB (86.6 GiB) and
28 to 29 GiB; in its 1M mode, 100,881 MiB, 16.1 GiB at ready and 12.9 at the lowest, at the end of a prompt of 985,296
tokens. The windows' KV cache is allocated at the start: a prompt does not grow it. What does grow is llama.cpp's cache
of the prompts it served, kept in host memory for the requests after (`--cache-ram`, 8 GiB by default): under GSM8K's
eight requests on the two windows it filled with entries of 1.5 to 5 GiB, llama-server held 11.9 GiB of host memory and
the box kept 6.0 GiB available at the lowest (5.6 under HumanEval in the 1M mode). The unit caps it at 2 GiB
(`--cache-ram 2048`): the same GSM8K run then kept 14.0 GiB available at the lowest, llama-server at 6.7 GiB, GSM8K 98.8 %
again, and the agent's five turns came from each window's own cache as before, 33 to 50 tokens computed again a turn
(2026-11-13). The 1M mode under the cap was not measured.

## A long prompt, the window, and an agent's next turn

A prompt of 30,196 tokens is computed at 815 tok/s (b12291; 787 under v0.9.0): 2.6 times slower than the Qwen3.8
Flash-Next FP4-Local lane's 2,141 tok/s at 32K. The next turn of the same conversation comes back from the cache: 31
tokens computed again, 30,192 kept. Every answer says how much of its prompt the cache served, with no option to set
(`usage.prompt_tokens_details.cached_tokens` on the OpenAI routes, `cache_read_input_tokens` on `/v1/messages`, read in
its `server-task.cpp`, b12291), as the other lanes say it with theirs.

The whole window is read (2026-11-12, b12291, one request): three codes hidden in a text of 255,529 tokens were all found,
the prompt computed at 676 tok/s, and the next turn came back from the cache with 23 tokens computed again. Past the
window the model was trained for, the server caps a request's context at 262,144 tokens unless YaRN is set ("the slot
context (524288) exceeds the training context of the model (262144) - capping"). With YaRN at twice it
(`FLASH_UNSLOTH_CTX=524288` and `FLASH_UNSLOTH_YARN=2` in the installer, one line each), three codes of three were found at
250,000 tokens and at 500,000 (the 513,367-token prompt computed at 584.6 tok/s, in 879 s), with 19 GiB available at
ready: an option, not the lane's, until its effect on short texts is measured. That was one request alone: `-c` holds a
window for each request, so with it set `FLASH_UNSLOTH_SLOTS=1`, since a second window of 262,144 tokens already takes
8.7 GiB more of the GPU's memory and two of 524,288 were not measured. The KV cache in FP8 is refused for this model
("FP8 cache is not supported for architecture 'qwen4exp'"); the installer takes llama.cpp's own f16 or q8_0. In q8_0
(2026-11-12, one request) the three codes were found at 250,000 tokens too, the prompt computed at 659.8 tok/s against
676.0 in f16 (2.4 % less), with 31 GiB available at ready against 27; f16 stays the lane's, since no evaluation of the
answers ran in q8_0.

At its native window the proxy keeps the flash model's ceiling of 250,000 tokens a prompt, and the Load gives opencode,
pi and `claude-local` the flash lane's limits, a context of 205,000 tokens and 32,000 for the answer. Its prompt cache
through the proxy, measured on its counters (`llamacpp:prompt_tokens_total` and `prompt_tokens_cached_total`, 2026-11-13):
an agent's five turns on a system prompt of 91,051 tokens computed it once (124 s to the first token in OpenAI's dialect,
128 s in Anthropic's), then 33 to 50 tokens a turn, the first token in 0.5 to 0.6 s; a Claude Code session and an opencode
session of the proof's own clients each took their second request from the cache (13,779 and 37,020 tokens).

## A window of 1,048,576 tokens for one request: FLASH_UNSLOTH_CONTEXT=1m

One request at a time with a window of 1,048,576 tokens, YaRN at four times and the KV cache in q8_0, `-c 1048576 -np
1`: the mode chosen beside the default on 2026-11-12 (decisions 1316 and 1318), as `FLASH_LOCAL_CONTEXT=1m` is the
FP4-Local lane's. Measured that day before 12:23, with no clock cap (b12291 built on the box, the DFlash2 draft alone):

| | measured |
|---|---|
| ready in | 51 s, 100,881 MiB of the GPU's |
| three codes hidden in a prompt of 985,296 tokens | 3 of 3 found, the prompt computed at 440.2 tok/s (2,240 s; the box's hottest zone read 89 °C at the end), the next turn back from the cache with 23 tokens computed again |
| memory available | 16.1 GiB at ready, 13.9 through the long prompt, 12.9 at the lowest, at its end |
| short texts at temperature 0 (code, reasoning, prose, a rewrite) | 58.2 / 62.1 / 31.4 / 51.6 tok/s, where one request at 262,144 in q8_0 gave 61.0 / 62.8 / 31.5 / 54.0 |

```bash
cd ~/dgx-spark-qwen38 && FLASH_UNSLOTH_CONTEXT=1m ./install-flash-unsloth.sh --units-only     # then Load the lane
cd ~/dgx-spark-qwen38 && FLASH_UNSLOTH_CONTEXT=native ./install-flash-unsloth.sh --units-only  # back to 262,144 each
```

`--units-only` writes the choice and the lane's unit, and downloads and starts nothing: the next Load of the lane, from
the cockpit or `./switch-model.sh flash-unsloth`, starts it with the window chosen and sets the clients' limits for it.
A Load of the lane while it serves without being the boot lane sets those limits alone (`./switch-model.sh flash-unsloth
--clients`): the boot, and opencode's default model with its compaction, stay with the lane the boot names.
The choice is kept for the Loads after (a Load from the cockpit stages the unit with none of your environment), in
`~/.config/qwen38/flash-unsloth-context`, which `--uninstall` removes. An update keeps the window chosen:
`./install.sh` does not read `FLASH_UNSLOTH_CONTEXT`, and says so.

At 1M the proxy relays a prompt up to 92 % of the window (964,689 tokens), and the Load gives opencode, pi and
`claude-local` a context of 830,000 tokens and 32,000 for the answer (the FP4-Local lane's rule at its 1M: the proxy's
share of the window, 92 %, less 10 % holds both, 830,000 + 32,000 = 862,000 <= 868,220), and the cockpit's Lanes and Now views say 1M tokens of context. A second request waits until
the first ends. Its answers under YaRN at four times: GSM8K 98.0 % (250 questions) and HumanEval 98.2 % (164 problems, pass@1) through the proxy (2026-11-13,
the Quality section).

## What it does not have

- No cache flush route and no route that aborts a generation: llama.cpp keeps and evicts its prompt cache by itself, and
  a client that closes its connection stops its generation (the proxy closes it, as on vLLM). The cockpit refuses Flush
  and Abort on this lane, in its name.
- No System One: its option labels are read off SGLang's top_logprobs, measured on the 27B and the flash lane alone. The
  proxy and the cockpit name those lanes.
- No vision: the GGUF's projector (`mmproj`) is not loaded, and its opencode and pi entries take text only.
- No load route: the cockpit reads its requests, the ones it defers and its draft's acceptance off `/metrics`, with the
  key. Its counters move when a request's prompt or answer is done, and are written with six significant digits.

## Installing, loading, removing

A plain install puts the lane's unit in, and its first Load from the cockpit pulls its image (4.5 GB, 2.7 to download)
and fetches its files (112 GB), then starts it; `--with-flash-unsloth` or `--all-weights` does all of it at install and
proves it serves, `--no-flash-unsloth` leaves it out of that run. From a terminal: `./prepare-lane.sh flash-unsloth`, then
`./switch-model.sh flash-unsloth`; `./switch-model.sh flash-unsloth --fetch` fetches its files alone. A 64 GB DGX Spark
does not hold it, and refuses it. `./install-flash-unsloth.sh --uninstall` removes its unit and its container, gives the
boot back to the text lane it took it from, and keeps its files and its image, said, for you to delete.
