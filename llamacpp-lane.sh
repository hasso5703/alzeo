# shellcheck shell=bash
# llamacpp-lane.sh: what the llama.cpp lanes' installers share (install-flash-unsloth.sh, the first; Unsloth's GGUF
# of the 27B is to come on it). Sourced after lanes.sh, never run. Each installer keeps what is its own (its pins,
# its unit's placeholders, its words) and names itself first, as a vLLM lane's does:
#
#   VLANE        the lane (flash-unsloth): its opencode provider, the record of the text lane it took the boot from
#                (lanes.sh), its --no-<lane>
#   VLANE_SHORT  the lane in a sentence, by its official name (Qwen3.8 Flash-Next Unsloth GGUF)
#   UNIT, INSTALLED, CONTAINER, CONFIG_DIR, HERE, LOCAL_IMAGE (the image, by its digest); and, for each part
#                below, what it says it reads.
#
# What vllm-lane.sh holds that is no vLLM's (the engine's cache, port and bind, the uninstall of the unit and of the
# clients' entries, the preflight, the room, the proof, and the way a rendered unit is put in place) is every text
# lane's on an engine of its own: it is sourced from there, not copied, since copies of it drifted when each vLLM
# installer had its own (vllm-lane.sh says how). What is the llama.cpp lanes' own is here: no key middleware (the
# server asks the key itself on every route but /health and /v1/health, read in its server-http.cpp), the GGUF files
# fetched with Python's standard library (gguf-fetch.py: the image has no Python), the check of those files at the
# unit's start (llamacpp-files.sh), the chat template a GGUF carries, and the engine's uninstall.

# shellcheck source=vllm-lane.sh
. "$HERE/vllm-lane.sh"

# ── The pinned files ─────────────────────────────────────────────────────────────────────────
# A GGUF file pinned by the installer as PATH=SHA256:SIZE (gguf-fetch.py's form): its blob and its link in the cache.
llane_blob(){  # $1 the repo, $2 PATH=SHA256:SIZE
  local rest="${2##*=}"
  printf '%s\n' "$HF_CACHE/hub/models--${1//\//--}/blobs/${rest%%:*}"
}
llane_path(){  # $1 the repo, $2 the revision, $3 PATH=SHA256:SIZE: where the snapshot links it, on the host
  printf '%s\n' "$HF_CACHE/hub/models--${1//\//--}/snapshots/$2/${3%=*}"
}
# The GB (10^9 bytes, as the Hub states sizes) of these files that are not whole in the cache yet, rounded up: what
# vlane_room is asked to find room for. The whole repository's blobs folder holds every quantization a user fetched
# there, which vlane_room's count of a repo's blobs would take off the need.
llane_missing_gb(){  # $1 the repo, $2... PATH=SHA256:SIZE
  local repo="$1" f blob size need=0
  shift
  for f in "$@"; do
    blob="$(llane_blob "$repo" "$f")"; size="${f##*:}"
    if [ ! -f "$blob" ] || [ "$(stat -c %s "$blob" 2>/dev/null || echo 0)" != "$size" ]; then
      need=$(( need + size ))
    fi
  done
  echo $(( (need + 999999999) / 1000000000 ))
}

# ── The fetch ────────────────────────────────────────────────────────────────────────────────
# The pinned files of one repository, at its pinned revision, into the cache (gguf-fetch.py says how): the cache
# first, so a Load on a box that holds them needs no network; resumed; checked against their sha256 as they come.
# The token by name only: gguf-fetch.py reads HF_TOKEN from its environment, so it is in no argv.
llane_fetch(){  # $1 the repo, $2 the revision, $3... PATH=SHA256:SIZE
  mkdir -p "$HF_CACHE"
  [ -z "${HF_TOKEN:-}" ] || export HF_TOKEN
  HF_HUB_DOWNLOAD_TIMEOUT=30 python3 -I "$HERE/gguf-fetch.py" "$HF_CACHE" "$@" \
    || die "the files of $1 could not be fetched (the line above says why). Causes: no internet, Hugging Face throttling an unauthenticated download (set HF_TOKEN=<your token>), a pinned revision removed upstream, or a permission error in $HF_CACHE. Re-running resumes."
}

# ── The chat template ────────────────────────────────────────────────────────────────────────
# The template the GGUF carries (its tokenizer.chat_template, read from its first part), with this lane's fixes
# (patch-template.py, the same fixes as the other Qwen lanes', the lean effort as the default among them), written to
# $1: again at every install and every switch, and left as it is, mtime included, when nothing changed. The rest is
# the lane's TEMPLATE_FIXES, VAR=value for patch-template.py's environment and --options for its command line, as
# vlane_template takes them.
llane_template(){  # $1 the file, $2 the GGUF's first part, then the lane's TEMPLATE_FIXES
  local out="$1" gguf="$2" a envs=() opts=()
  shift 2
  for a in "$@"; do
    case "$a" in --*) opts+=("$a") ;; *=*) envs+=("$a") ;; *) die "llane_template: $a is neither VAR=value nor --option" ;; esac
  done
  env ${envs[@]+"${envs[@]}"} python3 "$HERE/patch-template.py" ${opts[@]+"${opts[@]}"} "--gguf=$gguf" "$out" \
    || die "the chat template could not be patched (the lines above say why)"
}

# ── The service: the one part that needs root ────────────────────────────────────────────────
# The start scripts the unit runs before every start: the memory guard every lane runs, and the check of the files
# the engine reads. No key middleware: llama-server asks the key itself.
llane_service_files(){
  mkdir -p "$CONFIG_DIR"
  cmp -s "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh" \
    || install -m 755 "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh"
  cmp -s "$HERE/llamacpp-files.sh" "$CONFIG_DIR/llamacpp-files.sh" \
    || install -m 755 "$HERE/llamacpp-files.sh" "$CONFIG_DIR/llamacpp-files.sh"
}

# ── --uninstall ──────────────────────────────────────────────────────────────────────────────
# Its container and its image's local tag (lanes.sh: the image then merely dangling, which a prune reclaims), the
# check of its files unless another llama.cpp lane's unit runs it too; the checkpoints stay, said.
llane_uninstall_runtime(){  # $1... the checkpoints' repos
  local r d other
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  pin_untag "$VLANE"
  echo "the image ${LOCAL_IMAGE:0:40}... stays until a docker image prune (docker rmi it yourself if nothing else uses it)"
  other="$(grep -lsF 'llamacpp-files.sh' "$LANES_UNIT_DIR"/qwen38-*.service 2>/dev/null | grep -vxF "$INSTALLED" | head -1 || true)"
  if [ -n "$other" ]; then
    echo "llamacpp-files.sh kept: ${other##*/} runs it too"
  else
    rm -f "$CONFIG_DIR/llamacpp-files.sh"
  fi
  for r in "$@"; do
    d="$HF_CACHE/hub/models--${r//\//--}"
    [ -d "$d" ] && echo "checkpoint kept, delete it yourself for the space: rm -rf $d"
  done
  return 0
}
