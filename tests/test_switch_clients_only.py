#!/usr/bin/env python3
"""switch-model.sh <flash-local|flash-unsloth> --clients: the clients of a lane whose clients follow its unit's window
(opencode, and through its config pi and claude-local) set for the window its installed unit starts with, and nothing
else: no fetch, no unit rewritten, no boot moved, no engine touched, no record written.

A window chosen with FLASH_LOCAL_CONTEXT or FLASH_UNSLOTH_CONTEXT and --units-only reaches the unit and not the clients,
which switch-model.sh alone sets. The Load of such a lane that serves without being the boot lane, with that update
owed to it, restarted its unit alone: a switch would have pointed the boot at it, which no update asks for (2026-10-17),
and back from 1M the clients kept 840,000 tokens where the proxy refuses a prompt past 250,000. Its Load now runs this
before the restart (base.js, laneJourneySteps).

The script runs as written on a copy whose unit paths point at this test's files (test_switch_fetch_only.Box), with a
fake docker and a systemctl and a sudo that record every call and answer nothing.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_switch_fetch_only import Box  # noqa: E402

LANES = {
    # target: (its unit's engine flags at the 1M window, the generator of its provider, its served model, the 1M pair)
    "flash-local": ("vllm serve m --max-model-len 1048576 --max-num-seqs 8", "oc-flash-local.py",
                    "qwen3.8-flash-next-fp4", 840000),
    "flash-unsloth": ("-m /x.gguf -c 1048576 -np 1", "oc-flash-unsloth.py", "qwen3.8-flash-next-unsloth", 830000),
}


class ClientsOnly(unittest.TestCase):
    def box(self, target):
        box = Box()
        self.addCleanup(shutil.rmtree, box.t, True)
        cfg = box.home / ".config" / "qwen38"
        (cfg / "api-key").write_text("k\n")
        flags, gen, model, _ = LANES[target]
        unit = box.flocal_unit if target == "flash-local" else box.unsloth_unit
        unit.write_text(f"[Service]\nExecStart=/bin/bash -c '/usr/bin/docker run --rm --name qwen38-{target} x \\\n"
                        f"    {flags} --port 30000'\n")
        # the clients as the lane's 262,144-token window left them
        out = subprocess.run([sys.executable, str(REPO / gen), "http://127.0.0.1:30001/v1", str(cfg / "api-key"),
                              "205000", "32000", "a name (local, 262K)"], capture_output=True, text=True, check=True)
        base = {"model": "qwen38/qwen3.8-27b", "compaction": {"preserve_recent_tokens": 46000, "prune": False},
                "provider": {"qwen38": {"models": {"qwen3.8-27b": {
                    "name": "x", "limit": {"context": 547000, "input": 547000, "output": 182000}}}},
                    **json.loads(out.stdout)["provider"]}}
        (box.home / ".config" / "opencode").mkdir(parents=True)
        box.configs = (cfg / "opencode.json", box.home / ".config" / "opencode" / "opencode.json")
        for c in box.configs:
            c.write_text(json.dumps(base, indent=2) + "\n")
        # what the script runs beside it in this mode
        for f in ("oc-limits.sh", "oc-merge-limits.py", "oc-point-default.py", gen, "pi-gen.py", "boxmem.py"):
            (box.script.parent / f).write_text((REPO / f).read_text())
        (box.script.parent / "oc-limits.sh").chmod(0o755)
        box.target_unit = unit
        return box

    def test_the_clients_take_the_pair_of_the_units_window_and_nothing_else_changes(self):
        for target, (_, _, model, ctx) in LANES.items():
            with self.subTest(target=target):
                box = self.box(target)
                base_model, base_compaction = "qwen38/qwen3.8-27b", {"preserve_recent_tokens": 46000, "prune": False}
                units = {p: p.read_bytes() for p in box.target_unit.parent.iterdir()}
                rc, out, runs, calls = box.run(target, "--clients")
                self.assertEqual(rc, 0, out)
                for c in box.configs:
                    entry = json.loads(c.read_text())["provider"][target]["models"][model]
                    self.assertEqual(entry["limit"], {"context": ctx, "input": ctx, "output": 32000}, c)
                    self.assertTrue(entry["name"].endswith("(local, 1M)"), entry["name"])
                    # the boot stays on another lane: so do opencode's default model and its compaction, sized for it
                    doc = json.loads(c.read_text())
                    self.assertEqual((doc["model"], doc["compaction"]), (base_model, base_compaction), c)
                self.assertEqual(runs, [], "a docker run: a fetch")
                self.assertEqual({p: p.read_bytes() for p in box.target_unit.parent.iterdir()}, units, "a unit changed")
                self.assertEqual([c for c in calls if c.startswith("sudo ")
                                  or c.split()[0] in ("enable", "disable", "daemon-reload")], [], "the boot moved")
                self.assertFalse((box.home / ".config/qwen38/text-lane").exists(), "--clients wrote the text-lane record")
                self.assertNotIn("Switch queued", out)

    def test_another_target_is_refused_and_nothing_changes(self):
        box = self.box("flash-local")
        before = [c.read_bytes() for c in box.configs]
        rc, out, runs, calls = box.run("kolibri", "--clients")
        self.assertNotEqual(rc, 0)
        self.assertIn("--clients", out)
        self.assertEqual([c.read_bytes() for c in box.configs], before)
        self.assertEqual((runs, [c for c in calls if c.startswith("sudo ")]), ([], []))


if __name__ == "__main__":
    unittest.main()
