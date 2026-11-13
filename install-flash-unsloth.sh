#!/usr/bin/env bash
# Install the Qwen3.8 Flash-Next Unsloth GGUF lane: Qwen3.8-Flash-Next in Unsloth's dynamic GGUF quantization
# (unsloth/Qwen3.8-Flash-Next-GGUF, UD-Q4_K_XL: 76.9 GiB of weights, and the model's n-gram table, 26.8 GiB, which the
# engine reads from the disk as the tokens need it), drafted by ggml-org's DFlash2 head, served by llama.cpp's own
# server from the image its project publishes, on the text lanes' port behind the proxy and its key. Idempotent, and
# safe to run alone.
#
#   ./install-flash-unsloth.sh                 first install, or update in place
#   ./install-flash-unsloth.sh --no-smoke      skip the proof at the end
#   ./install-flash-unsloth.sh --uninstall     remove the unit (the GGUF files and the image kept, said)
#   ./install-flash-unsloth.sh --units-only    the unit alone: no image, no checkpoint (install.sh, on demand)
#   ./install-flash-unsloth.sh --prepare       its image, pulled with no sudo: the cockpit's first Load
#                                              runs it (./prepare-lane.sh flash-unsloth)
#   ./install-flash-unsloth.sh --fetch         its GGUF files and its draft, into the cache, with no sudo:
#                                              switch-model.sh flash-unsloth runs it before --stage
#   ./install-flash-unsloth.sh --stage FILE    the unit for the files in the cache, written to FILE with no
#                                              sudo, its chat template made first: switch-model.sh
#                                              flash-unsloth installs FILE
#
# THE WINDOW is 262,144 tokens, the model's own, for each of two requests at once: FLASH_UNSLOTH_CONTEXT=1m gives one
# request at a time a window of 1,048,576 (YaRN at four times) instead, and the choice holds for every run after, a
# Load from the cockpit included (FLASH_UNSLOTH_CONTEXT=native brings the default back), as FLASH_LOCAL_CONTEXT
# does for the FP4-Local lane. A window shared by several requests (-kvu) is refused for now: FLASH_UNSLOTH_SHARED says
# why.
#
# install.sh puts this lane's unit in on every plain install, like the other lanes, and its first Load from the
# cockpit pulls its image and fetches its files; ./install.sh --no-flash-unsloth leaves it out of the run that passes
# it. It takes 112 GB of GGUF files (the model's four parts, 111.3 GB, and the draft, 0.7) and an image of 4.5 GB
# (2.7 to download): when the disk does not have the room, this refuses before downloading anything.
#
# WHY A LANE OF ITS OWN. The flash lane serves the same model on SGLang, from NVFP4 checkpoints, and the FP4-Local lane
# on vLLM, from NVIDIA's FP4 build. This one is Unsloth's GGUF on llama.cpp: another engine and another quantization
# of the same weights, whose n-gram table stays on the disk (--lazy-mode on: the unit says why), with an n-gram lookup
# and a DFlash2 draft. llama.cpp's image is its project's own, for arm64 with CUDA, pulled by digest: nothing is built on
# the box (llama.cpp's speech lane builds its own server for the CPU instead, install-speech.sh). The GGUF files are
# pinned one by one, by their sha256 and their size, and fetched with gguf-fetch.py: the image has no Python, and the
# repository holds every quantization of the model, of which the lane takes one.
#
# ONE ENGINE AT A TIME. The unit says so with Conflicts=: starting it stops whichever lane serves, and starting
# another lane stops it.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-flash-unsloth.service
# where systemd reads the units, the one variable lanes.sh and llamacpp-lane.sh read the other units by
INSTALLED="${LANES_UNIT_DIR:-/etc/systemd/system}/$UNIT"
CONTAINER=qwen38-flash-unsloth
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
step(){ printf '\n\033[1;36m── %s\033[0m\n' "$*"; }

# Under sudo every path below moves to /root: the cache, the config and the key the unit
# reads. install.sh refuses the same way, for the same reason.
if [ "$(id -u)" = "0" ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
  die "run this as the user who will use the box, not as root (everything it installs is addressed from \$HOME)."
fi

SMOKE=1; ACTION=install; MODE=full
for a in "$@"; do case "$a" in
  --no-smoke) SMOKE=0 ;;
  --uninstall) ACTION=uninstall ;;
  # The two halves of a lane installed on demand, as for the other lanes: --units-only puts in
  # what needs root (the unit, its start scripts) and nothing heavy, and --prepare pulls the
  # lane's image as the user, with no sudo, when the cockpit first loads the lane. The GGUF files
  # come with that Load's switch (--fetch), as a checkpoint comes with any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  --fetch) [ "$MODE" = full ] || die "--fetch is a mode of its own"; MODE=fetch ;;
  --stage) [ "$MODE" = full ] || die "--stage is a mode of its own"; MODE=stage ;;
  -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
  *) if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then STAGE_OUT="$a"; else die "unknown option: $a"; fi ;;
esac; done
if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then die "--stage takes the file to write the unit to"; fi

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"
# A 64 GB DGX Spark cannot hold this lane (boxmem.py): refused there before anything is fetched,
# and --uninstall still takes away what an older release put there.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits flash-unsloth || die "The Qwen3.8 Flash-Next Unsloth GGUF lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the llama.cpp lanes' installers share: the engine's cache and port, the uninstall, the service, the
# preflight and the room, the fetch of the GGUF files, the chat template, the proof (part of it vllm-lane.sh's)
# shellcheck source=llamacpp-lane.sh
. "$HERE/llamacpp-lane.sh"
VLANE=flash-unsloth; VLANE_SHORT="Qwen3.8 Flash-Next Unsloth GGUF"

# The pins, at column 0 like install.sh's. switch-model.sh, check-pins.sh and the cockpit read them here.
# Unsloth's GGUF of the model at its revision of 2026-11-11, whose UD-Q4_K_XL folder holds the four parts the lane
# serves, each by its sha256 and its size in bytes (PATH=SHA256:SIZE, gguf-fetch.py's form): the first carries the
# header and the chat template, the second and third the weights (76.9 GiB), the fourth the n-gram table in IQ4_NL
# (26.8 GiB), read from the disk. Read on the reference box from the cache's own blobs (huggingface_hub names a file
# kept in LFS by its sha256) and each the same in the Hub's LFS pointer at the revision, 2026-11-12 (check-pins.sh
# asks the pointers every day).
FLASH_UNSLOTH_REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
FLASH_UNSLOTH_REV="d96201194f103c906994f283f504209dc8d2a10c"
FLASH_UNSLOTH_PART1="UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf=acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858:10946618"
FLASH_UNSLOTH_PART2="UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf=45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4:49720539552"
FLASH_UNSLOTH_PART3="UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf=8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82:32803030176"
FLASH_UNSLOTH_PART4="UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf=5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36:28800138432"
# The draft, ggml-org's GGUF of the DFlash2 head of incoai (Q8_0): measured against the MTP head and the DFlash2 head
# in Q4_0, the one that decodes fastest at the model card's sampling (docs/flash-unsloth-lane.md).
FLASH_UNSLOTH_DRAFT_REPO="ggml-org/Qwen3.8-Flash-Next-GGUF"
FLASH_UNSLOTH_DRAFT_REV="b9ad2d241b433afd84a423760ccf11929c9a71fa"
FLASH_UNSLOTH_DRAFT="dflash-Qwen3.8-Flash-Next-Q8_0.gguf=4670408ccf79cf36b5f2cad2d6f214d0b47b93b1ff693c3a5bc68126007e64e2:652101152"
# llama.cpp's server image for arm64 with CUDA 13 (server-cuda13: b12316, commit 5d4c8eeb2), by its digest: the arm64
# image of the tag server-cuda13-b12316, created on 2026-11-12, its labels (version b12316, revision 5d4c8eeb24db) and
# the tag's index read on the reference box. It asks CUDA 13.5 or a driver of the 535, 580, 595, 610 or 615 branch (its
# NVIDIA_REQUIRE_CUDA), which the box's 580.178.04 meets. It replaced b12282's image, measured first, on which the n-gram
# lookup slowed the DFlash2 draft down (SPEC below, 2026-11-12).
FLASH_UNSLOTH_IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:a5600a59c704208491adbb96505454a8d3d4cd3a9db48b03cf2760b660b7fd16"
# the files at these revisions, 111.99 GB, rounded up: the room their fetch asks, here and in switch-model.sh
FLASH_UNSLOTH_GB="112"
# What drafts (--spec-type): llama.cpp's n-gram lookup (ngram-mod) before ggml-org's DFlash2 head. On the pinned image,
# at the model card's sampling, one request, the weights copied into the GPU's memory from the USB SSD, each text's first
# pass (the lookup's pool keeps what a pass wrote, so a second pass of the same text is no measure of it): code 51.8
# tok/s, reasoning 56.9, prose 31.5, a rewrite of code 123.7 (variantes-20261112-0840-choisi-tiede-image-b12316). On
# b12282's image, the same way the same morning (variantes-20261112-0727-choisi-image-ssd), the pair gave 46.9, 43.9,
# 22.9 and 116.2, and the DFlash2 head alone 53.8, 58.8, 31.3 and 52.0 (the median of three passes): hence b12316.
# The head alone on b12316 came after a slow start (261 s) and its first passes were slow (34.2 tok/s in code): no
# witness. b12291 built on the box paired them as well (variantes-20261112-0056-choisi-b12291, in mmap: 62.4, 52.0,
# 29.6 and 121.0, against 56.1, 54.7, 29.4 and 48.2 with the head alone). draft-dflash alone is the head without the
# lookup.
FLASH_UNSLOTH_SPEC="ngram-mod,draft-dflash"
# The options measured on the night of 2026-11-11 to 12 (llama.cpp b12291 built on the box, one variable each; the unit
# takes them as they are, and render_unit refuses a value the engine would not take):
# How the weights are read (--load-mode), the n-gram table staying on the disk in every mode (--lazy-mode on: b12291
# maps a lazy tensor whatever the load mode, src/llama-model-loader.cpp init_mappings, and so does the pinned b12316,
# 25 commits after it, none in the loader). With mmap the GPU reads the weights in place in the page cache: from the
# same files it decoded at 27.9 tok/s in the evening's runs and at about 25 in the night's; with none they are copied
# into its memory. Side by side, the weights on the internal disk, no draft, temperature 0, the mean of four probes
# (variantes-20261112-0351-modes-chargement-poids-nvme): mmap 24.9 tok/s, ready in 273 s; auto 27.3, 40 s; none 27.6,
# 35 s; dio 27.5, 25 s (the three after mmap with its weights in the page cache, dio past it); 30 to 32 GiB available
# at ready in all four. The pinned image serves in none: from the USB SSD it was ready in 261 s, then 50 s for the start
# after it on the same files (variantes-20261112-0840-choisi-tiede-image-b12316).
FLASH_UNSLOTH_LOAD="none"
# The requests served at once (-np): 2, each with a window of its own of 262,144 tokens (SHARED below), the KV cache in
# llama.cpp's own f16, as Hasan chose (decision 1318, 2026-11-12; FLASH_UNSLOTH_CONTEXT=1m below is the mode beside it).
# Measured on 2026-11-12 with b12291 built on the box, the weights on the internal disk and the DFlash2 head alone
# (variantes-20261112-0850-flux-places-b12291, np2-c524): ready in 50 s, 97,552 MiB of the GPU's, 20 GiB available at
# ready and 14.2 at the lowest under its probes (the box's telemetry); 54.2 tok/s for one request, 63.2 in all with two
# (34.2 each), 66.9 with four (two served, two waiting); an agent's five turns back from the cache with 61 to 77 tokens
# computed again after the first. In q8_0 on the pinned image b12316 (variantes-20261112-1501-kvu-image-image-b12316,
# np2-c524-q8-image): four salvoes of 4 and 8 requests, none stopped (63.2 to 65.3 tok/s in all under a GPU clock cap of
# 2,200 MHz, which every measure of that afternoon ran under), two conversations taking turns with 19 and 20 tokens
# computed again, 24 GiB available at ready and 16.1 at the lowest. Four windows in q8_0 (-c 1048576 -np 4, -0850-):
# 53.0 tok/s for one request, 68.9 with four (18.2 each), 14 GiB at ready and 10 after, too thin a margin for the box.
FLASH_UNSLOTH_SLOTS="2"
# How they hold their windows: 0, a window of CTX for each, -c being then CTX times SLOTS, the lane's: no request can
# make another fail. 1, one window of CTX they share (-kvu), is refused (decision 1318): it was meant to serve more
# requests at once on more context, and llama-server stopped itself under load in four of the nine runs that shared a
# window on 2026-11-12, with b12291 built on the box and with the pinned image b12316 ("common/common.cpp:1634: failed
# to remove sequence N with p0=..., p1=-1", an abort in common_memory::seq_rm from post_decode: nuit-20261112-1412 and
# -1454, variantes-20261112-1448-places-524-cram4-b12291 and -1501-kvu-image-image-b12316). The five others, salvos of
# 1, 4 and 8 short requests (variantes-20261112-0415-flux, -1304 and -1327), ended well: the abort is intermittent. It
# is llama.cpp's open issue 31656: a draft's rejected tokens are rolled back from this hybrid model's recurrent snapshots,
# which a sequence loses when find_slot moves its cell between the parts of a batch that a shared window splits out of
# the sequences' order. Before they stopped: four sharing 524,288 by YaRN at twice, 22 GiB available at ready, 53.2
# tok/s for one request and 72.6 in all with four under the clock cap of 2,200 MHz; four sharing 1,048,576, 11 GiB at
# ready, then under 6. It comes back when llama.cpp fixes it (docs/flash-unsloth-lane.md). A setting of this installer,
# as SLOTS and CTX are: the environment does not set it.
FLASH_UNSLOTH_SHARED="0"
# A request's window, the context the model was trained for (qwen4exp.context_length in the GGUF): past it the server
# caps a slot at 262,144 without YaRN. YARN, YaRN's factor over it, empty for none: with 2 (CTX 524288) three codes of
# three were found at 250,000 and at 500,000 tokens, the 513,367-token prompt computed at 584.6 tok/s
# (variantes-20261112-0429-contexte-b12291), one request alone, the KV cache in f16, 19 GiB available at ready. -c holds
# CTX for each request: with a CTX past 262,144 set SLOTS to 1, since a second window of 262,144 tokens took 8.7 GiB
# more of the GPU's memory (88,693 MiB with one, 97,552 with two, b12291) and two of 524,288 were not measured.
# FLASH_UNSLOTH_CONTEXT=1m (below) gives one request 1,048,576.
FLASH_UNSLOTH_CTX="262144"
FLASH_UNSLOTH_YARN=""
# The KV cache's type (-ctk and -ctv): empty for llama.cpp's own f16, the lane's (its GSM8K and HumanEval were measured
# in f16: the lane's page, Quality), or q8_0, which the 1M mode takes; FP8 is refused for this architecture ("FP8 cache
# is not supported for architecture 'qwen4exp'", variantes-20261112-0429-contexte-b12291). q8_0 at 262,144, one request
# (variantes-20261112-0704-contexte-q8-b12291): three codes of three at 250,000 tokens, that prompt computed at 659.8
# tok/s against 676.0 in f16 (2.4 % less), 31 GiB available at ready against 27, its short decodes at temperature 0 at
# 61.0 / 62.8 / 31.5 / 54.0 tok/s. Its answers' quality in q8_0 was measured with YaRN at four times, in the 1M mode below.
FLASH_UNSLOTH_KV=""
# FLASH_UNSLOTH_CONTEXT=1m: one request at a time with a window of 1,048,576 tokens, YaRN at four times, its KV cache in
# q8_0 whatever KV says (the mode beside the default, decisions 1316 and 1318), as FLASH_LOCAL_CONTEXT=1m is the
# FP4-Local lane's. Measured (variantes-20261112-1001-contexte-1m-b12291, b12291 built on the box, the DFlash2 head
# alone): ready in 51 s, 100,881 MiB of the GPU's, 16.1 GiB available at ready, 13.9 through a prompt of 985,296 tokens
# and 12.9 at the lowest, at its end (the box's telemetry); three codes of three found in it, computed at 440.2 tok/s;
# short texts decoded at temperature 0 at 58.2 / 62.1 / 31.4 / 51.6 tok/s, where one request at 262,144 in q8_0 gave
# 61.0 / 62.8 / 31.5 / 54.0 (variantes-20261112-0704-contexte-q8-b12291). Its answers through the proxy: GSM8K 98.0 % and
# HumanEval 98.2 % (the lane's second proof, 2026-11-13; at the native window GSM8K 98.8 % in that proof, HumanEval 99.4 %
# in the study). Kept for the runs after in a file of its own: a Load from the cockpit stages the unit with none of the
# user's environment. native (or empty) brings the default back. The choice is read past the uninstall, which removes
# it, and kept by the render that takes it, and by --prepare.
CONTEXT_CHOICE="$CONFIG_DIR/flash-unsloth-context"
MODEL="$FLASH_UNSLOTH_REPO"; MODEL_REV="$FLASH_UNSLOTH_REV"
MODEL_FILES=("$FLASH_UNSLOTH_PART1" "$FLASH_UNSLOTH_PART2" "$FLASH_UNSLOTH_PART3" "$FLASH_UNSLOTH_PART4")
DRAFT="$FLASH_UNSLOTH_DRAFT_REPO"; DRAFT_REV="$FLASH_UNSLOTH_DRAFT_REV"; DRAFT_FILE="$FLASH_UNSLOTH_DRAFT"
LOCAL_IMAGE="$FLASH_UNSLOTH_IMAGE"
# Its pull at its peak: 4.50 GB unpacked and 2.70 GB to download (docker image inspect and docker manifest inspect
# on the reference box, 2026-11-12), which the containerd image store keeps beside it, 6.7 GiB; ceil(x 1.02) + 1 as
# df -BG rounds up (install-kolibri.sh counts its image so).
IMAGE_GB=8

# The text lanes' cache and port: this lane is one of them for the proxy (vllm-lane.sh).
vlane_engine_env "${FLASH_UNSLOTH_PORT:-}"

# The chat template the GGUF carries, with the Qwen lanes' fixes (llamacpp-lane.sh, patch-template.py): Unsloth's
# template merges the system messages that open a conversation and still refused one past them, which opencode sends
# (a 400 on every request of the Ornith lane for the same reason, 2026-10-24), and raised on a Claude Code ToolSearch
# result; and the lean effort is its default, as on the other Qwen lanes (LEAN.md): the GGUF's is xhigh.
TEMPLATE="$CONFIG_DIR/chat-template-flash-unsloth.jinja"
TEMPLATE_FIXES=(LEAN_DEFAULT=1)
part1_host(){ llane_path "$MODEL" "$MODEL_REV" "$FLASH_UNSLOTH_PART1"; }
lane_template(){ llane_template "$TEMPLATE" "$(part1_host)" "${TEMPLATE_FIXES[@]}"; }
lane_template_if_cached(){ [ -f "$(part1_host)" ] || return 0; lane_template; }

if [ "$ACTION" = uninstall ]; then
  step "Removing the Qwen3.8 Flash-Next Unsloth GGUF lane"
  vlane_uninstall_unit
  vlane_uninstall_clients
  # The image is pulled, not built: its local tag goes (lanes.sh), and the GGUF files stay, said.
  llane_uninstall_runtime "$MODEL" "$DRAFT"
  rm -f "$TEMPLATE" "$CONTEXT_CHOICE"
  exit 0
fi

# FLASH_UNSLOTH_CONTEXT (above): given, checked and kept by the render; else the choice kept before, or the default.
CONTEXT_GIVEN=""
if [ -n "${FLASH_UNSLOTH_CONTEXT+set}" ]; then
  case "$FLASH_UNSLOTH_CONTEXT" in
    1m) ;;
    ""|native) FLASH_UNSLOTH_CONTEXT=native ;;
    *) die "FLASH_UNSLOTH_CONTEXT=$FLASH_UNSLOTH_CONTEXT: native (262,144 tokens for each of 2 requests at once, the default) or 1m (1,048,576 for one request)" ;;
  esac
  CONTEXT_GIVEN=1
elif [ "$(head -1 "$CONTEXT_CHOICE" 2>/dev/null)" = 1m ]; then
  FLASH_UNSLOTH_CONTEXT=1m
else
  FLASH_UNSLOTH_CONTEXT=native
fi
# The choice given, kept for the runs after (a Load stages the unit with none of the user's environment).
keep_choice(){
  [ -n "$CONTEXT_GIVEN" ] || return 0
  if [ "$FLASH_UNSLOTH_CONTEXT" = 1m ]; then mkdir -p "$CONFIG_DIR" && echo 1m > "$CONTEXT_CHOICE"; else rm -f "$CONTEXT_CHOICE"; fi
}

# The unit, rendered for this box: written to $1. Its start checks the files the engine reads (llamacpp-files.sh):
# the four parts, the draft and the chat template, each where the unit names it.
render_unit(){
  local n="$FLASH_UNSLOTH_SLOTS" c="$FLASH_UNSLOTH_CTX" factor="$FLASH_UNSLOTH_YARN" type="$FLASH_UNSLOTH_KV" kv="" yarn=""
  local slots="" ctx="" files="" f
  local in_box="/root/.cache/huggingface"
  # Numbers written with no leading zero: bash's arithmetic reads one as octal, and 08 failed ("value too great for
  # base") where 0262144 rendered -c 182472 with no word said (code review of v1.48.0, 2026-11-12).
  case "$FLASH_UNSLOTH_SLOTS" in ''|*[!0-9]*|0*) die "FLASH_UNSLOTH_SLOTS must be a number of requests, from 1, with no leading zero, got: $FLASH_UNSLOTH_SLOTS" ;; esac
  case "$FLASH_UNSLOTH_CTX" in ''|*[!0-9]*|0*) die "FLASH_UNSLOTH_CTX must be a number of tokens, with no leading zero, got: $FLASH_UNSLOTH_CTX" ;; esac
  case "$FLASH_UNSLOTH_LOAD" in mmap|auto|none|dio) ;; *) die "FLASH_UNSLOTH_LOAD must be mmap, auto, none or dio, got: $FLASH_UNSLOTH_LOAD" ;; esac
  case "$FLASH_UNSLOTH_YARN" in '') ;; *[!0-9]*|0*|1) die "FLASH_UNSLOTH_YARN must be empty or YaRN's factor, 2 or more, with no leading zero, got: $FLASH_UNSLOTH_YARN" ;; esac
  case "$FLASH_UNSLOTH_KV" in ''|f16|q8_0) ;; *) die "FLASH_UNSLOTH_KV must be empty, f16 or q8_0 (FP8 is refused for this model), got: $FLASH_UNSLOTH_KV" ;; esac
  case "$FLASH_UNSLOTH_SHARED" in
    0) ;;
    1) die "FLASH_UNSLOTH_SHARED=1 (one window the requests share, -kvu) is refused: llama-server stopped itself under load in four of the nine runs that shared a window (\"failed to remove sequence\", measured on 2026-11-12 with b12291 and with the image b12316; llama.cpp issue 31656). Set it to 0, a window each." ;;
    *) die "FLASH_UNSLOTH_SHARED must be 0 (a window for each request), got: $FLASH_UNSLOTH_SHARED" ;;
  esac
  # FLASH_UNSLOTH_CONTEXT=1m: one request, a window of 1,048,576 by YaRN at four times, its KV cache in q8_0, as measured
  # (above)
  if [ "$FLASH_UNSLOTH_CONTEXT" = 1m ]; then n=1; c=1048576; factor=4; type=q8_0; fi
  # a window for each request, -c holding them all
  slots="-np $n"; ctx=$((c * n))
  [ -z "$type" ] || kv="-ctk $type -ctv $type"
  [ -z "$factor" ] || yarn="--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale $factor"
  for f in "${MODEL_FILES[@]}"; do files="$files $(llane_path "$MODEL" "$MODEL_REV" "$f")"; done
  files="$files $(llane_path "$DRAFT" "$DRAFT_REV" "$DRAFT_FILE") $TEMPLATE"
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$LOCAL_IMAGE|g" -e "s|__HF_CACHE__|$HF_CACHE|g" -e "s|__FILES__|${files# }|g" \
      -e "s|__MODEL__|$in_box/hub/models--${MODEL//\//--}/snapshots/$MODEL_REV/${FLASH_UNSLOTH_PART1%=*}|g" \
      -e "s|__DRAFT__|$in_box/hub/models--${DRAFT//\//--}/snapshots/$DRAFT_REV/${DRAFT_FILE%=*}|g" \
      -e "s|__CTX__|$ctx|g" -e "s|__SLOTS__|$slots|g" -e "s|__KV__|$kv|g" -e "s|__SPEC__|$FLASH_UNSLOTH_SPEC|g" \
      -e "s|__LOAD__|$FLASH_UNSLOTH_LOAD|g" -e "s|__YARN__|$yarn|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
  keep_choice
}

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a lane
# installed on demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  llane_service_files
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  # with what the engine reads at its start besides the unit, its chat template: a lane started before it
  # changed serves the old one (vllm-lane.sh)
  vlane_put_unit "$RENDER" "$TEMPLATE"
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Qwen3.8 Flash-Next Unsloth GGUF in the Lanes view"
  echo "  a terminal : ./switch-model.sh flash-unsloth, then the commands it prints"
}

# The room the files still missing need on the cache's disk, said before a byte is downloaded: the GGUF files and
# the draft together, so that the draft never finds the disk full after the parts came.
missing_gb(){ echo $(( $(llane_missing_gb "$MODEL" "${MODEL_FILES[@]}") + $(llane_missing_gb "$DRAFT" "$DRAFT_FILE") )); }
fetch_files(){
  llane_fetch "$MODEL" "$MODEL_REV" "${MODEL_FILES[@]}"
  llane_fetch "$DRAFT" "$DRAFT_REV" "$DRAFT_FILE"
}

# --fetch: the files, for switch-model.sh, with no sudo, the room checked first.
if [ "$MODE" = fetch ]; then
  need_gb="$(missing_gb)"
  if [ "$need_gb" -gt 0 ]; then
    need_gib=$(( (need_gb * 1000000000 + 1073741823) / 1073741824 )); free_gib="$(free_gb_at "$HF_CACHE")"
    [ "${free_gib:-0}" -ge "$need_gib" ] \
      || die "${free_gib:-?} GiB free under HF_CACHE=$HF_CACHE, the GGUF files need about $need_gib GiB more. Free some space first: nothing was downloaded."
  fi
  fetch_files
  exit 0
fi

# --stage: the unit for the files in the cache, for switch-model.sh, with no sudo: its template first, as the unit
# names it. This lane's own files only, as the other lanes' switches touch nothing but their unit: the start scripts
# every lane runs are ./install.sh's to put in (the Qwen3.8 Flash-Next FP4-Local lane's rule), but for the check of
# this lane's files when it is missing.
if [ "$MODE" = stage ]; then
  lane_template >&2
  [ -f "$CONFIG_DIR/llamacpp-files.sh" ] || install -m 755 "$HERE/llamacpp-files.sh" "$CONFIG_DIR/llamacpp-files.sh"
  render_unit "$STAGE_OUT"
  grep -q '__[A-Z][A-Z0-9_]*__' "$STAGE_OUT" && die "the unit template still holds an unsubstituted placeholder"
  exit 0
fi

step "1/5 Preflight (Qwen3.8 Flash-Next Unsloth GGUF: $MODEL @ ${MODEL_REV:0:12})"
vlane_preflight engine-preflight.sh llamacpp-files.sh gguf-fetch.py patch-template.py
# what is left to download of the pinned files, not what the repositories' folders hold (llamacpp-lane.sh)
WEIGHTS_GB="$(missing_gb)"
vlane_room "$LOCAL_IMAGE"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the GGUF files come with the lane's first Load"
  lane_template_if_cached
  install_service
  step "Done: the Qwen3.8 Flash-Next Unsloth GGUF lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Qwen3.8 Flash-Next Unsloth GGUF in the Lanes view pulls its image (about 4.5 GB, 2.7 to"
  echo "               download), fetches its GGUF files (about $FLASH_UNSLOTH_GB GB), then starts it"
  echo "  a terminal : ./install-flash-unsloth.sh --prepare, then ./switch-model.sh flash-unsloth and the commands it prints"
  exit 0
fi

step "2/5 Image (${FLASH_UNSLOTH_IMAGE:0:40}...)"
if docker image inspect "$FLASH_UNSLOTH_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$FLASH_UNSLOTH_IMAGE" \
    || die "docker pull of $FLASH_UNSLOTH_IMAGE failed. Causes: no internet, a GitHub Container Registry rate limit (retry in a few minutes), or the digest removed upstream."
fi
# its local tag, so that a docker image prune leaves it alone, and the tags of the images it was pinned to before
# go (lanes.sh)
pin_tag flash-unsloth "$FLASH_UNSLOTH_IMAGE"
unpin_stale flash-unsloth "$FLASH_UNSLOTH_IMAGE" || true

if [ "$MODE" = prepare ]; then
  # its Load renders the unit, with the window chosen here (FLASH_UNSLOTH_CONTEXT)
  keep_choice
  step "Done: the Qwen3.8 Flash-Next Unsloth GGUF lane's image is ready; its Load fetches the GGUF files and starts it"
  exit 0
fi

step "3/5 GGUF files ($MODEL @ ${MODEL_REV:0:12} and its draft, resumable) and the chat template"
fetch_files
lane_template

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
VLANE_SERVED=qwen3.8-flash-next-unsloth; VLANE_KWARGS='{"enable_thinking":false}'
VLANE_PARSERS="llama.cpp's parser of the template's tool calls is"; VLANE_BOOT="about 4 to 7 min"; VLANE_WAIT_MIN=40
vlane_prove

step "Done: the Qwen3.8 Flash-Next Unsloth GGUF lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Qwen3.8 Flash-Next Unsloth GGUF in the cockpit, or ./switch-model.sh flash-unsloth"
echo "  clients      : the proxy on :30001 with the box's key, model qwen3.8-flash-next-unsloth (OpenAI and Anthropic APIs)"
