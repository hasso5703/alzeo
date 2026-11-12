#!/usr/bin/env python3
"""The Qwen3.8 Flash-Next Unsloth GGUF lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the two
cannot drift.

Usage: oc-flash-unsloth.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"flash-unsloth": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# The GGUF's chat template is Qwen's with Unsloth's changes, patched as the other Qwen lanes' are
# (patch-template.py): the same four levels, lean first, the template's default, as install.sh lists
# them for the 27B and the flash lane, so that a picker shows first the level a request that names none
# gets. llama.cpp hands chat_template_kwargs to the template as they come (its server-common.cpp).
LEVELS = ("lean", "low", "medium", "xhigh")


def provider(base_url, key_file, ctx, out, name):
    """The provider block. Text only: the lane loads no vision projector (the GGUF's mmproj file is not
    served), so an image sent to it would reach a model that cannot read it."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Qwen3.8-Flash-Next Unsloth GGUF (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"qwen3.8-flash-next-unsloth": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: {"chat_template_kwargs": {"reasoning_effort": lvl}} for lvl in LEVELS},
            "attachment": False,
            "modalities": {"input": ["text"], "output": ["text"]},
        }},
    }


def main():
    if len(sys.argv) != 6:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    base_url, key_file, ctx, out, name = sys.argv[1:6]
    try:
        ctx, out = int(ctx), int(out)
    except ValueError:
        print("ctx and out are token counts", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"provider": {"flash-unsloth": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
