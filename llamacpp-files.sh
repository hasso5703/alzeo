#!/bin/bash
# llamacpp-files.sh <unit> <target> <file>...: what a llama.cpp lane's engine reads has to be there before it
# runs, and its unit asks this before the engine, as its main process (the head of its ExecStart, after the
# memory guard every lane runs), as flash-local-files.sh is the Qwen3.8 Flash-Next FP4-Local lane's.
#
# The files are the GGUF parts and the draft the unit names, each a link of the Hugging Face cache's snapshot
# to its blob (gguf-fetch.py), and the chat template it mounts: a link whose blob is not there, or an empty
# file, is no file. Without them
# llama-server stops at its load, and systemd starts it again every 15 s; and docker, for a cache on a disk
# not mounted yet, would make the folders it mounts, owned by root (flash-local-files.sh says how that was
# found). A cache on a disk mounted late at boot gets 90 s to come, counted in tries rather than read off the
# clock, which can jump at boot. Without them this says what is missing and what brings it back, the disk
# first, on a line the cockpit reads ("preflight: <unit> lacks its files"), and exits 78, which the unit's
# RestartPreventExitStatus holds for its main process.
#
# A script of its own: engine-inputs.py leaves it out of what the engine reads (NOT_INPUTS), so a new wording
# of it never asks a serving lane to restart. LLAMACPP_FILES_TRIES sets the tries of 3 s (30 by default, the
# tests' to shorten).
set -u
unit="${1:?usage: llamacpp-files.sh <unit> <target> <file>...}"; target="${2:?usage: llamacpp-files.sh <unit> <target> <file>...}"
shift 2
tries="${LLAMACPP_FILES_TRIES:-30}"
[ "$#" -gt 0 ] || { echo "preflight: $unit lacks its files: its unit names none (re-run ./install.sh)" >&2; exit 78; }

# The first of the files given that is missing or empty, said on stdout; nothing when all are there.
first_missing() {
  local f
  for f in "$@"; do
    if [ ! -f "$f" ] || [ ! -s "$f" ]; then
      echo "$f"; return
    fi
  done
}

missing=""
for ((t = 0; t <= tries; t++)); do
  missing="$(first_missing "$@")"
  [ -z "$missing" ] && exit 0
  [ "$t" -lt "$tries" ] && sleep 3
done
echo "preflight: $unit lacks its files: $missing is missing or empty (a GGUF part or the draft), or the disk" \
     "that holds the Hugging Face cache is not mounted. With that disk mounted, Load the lane from the cockpit," \
     "or run ./switch-model.sh $target" >&2
exit 78
