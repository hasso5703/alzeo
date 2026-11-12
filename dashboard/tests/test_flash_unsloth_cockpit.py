#!/usr/bin/env python3
"""The cockpit follows the Qwen3.8 Flash-Next Unsloth GGUF lane on its own facts, the first lane on llama.cpp (v1.48).

Its engine is llama.cpp's own server: its /health open and generating nothing, every other route behind the box's key
(/v1/models, /props and /metrics among them), none of SGLang's or vLLM's routes. The engine here answers as llama-server's
source writes its answers (b12291: server-context.cpp's /v1/models and /props, server-task.cpp's /metrics), asking the
key as its server-http.cpp asks it; its boot is the log of llama.cpp's image b12282, measured before b12316, on the
reference box (fixtures/flash-unsloth-boot.log). TODO(measure): its /v1/models, /props and /metrics captured from the lane's unit, as
the vLLM lanes' fixtures are. The module is imported with its environment pointed at a throwaway box.
"""
import http.server
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
BOOT = (HERE.parent / "fixtures" / "flash-unsloth-boot.log").read_text().splitlines()
U = "qwen38-flash-unsloth.service"
REPO_ID = "unsloth/Qwen3.8-Flash-Next-GGUF"
SERVED = "qwen3.8-flash-next-unsloth"
MODELS = {"object": "list", "data": [{"id": SERVED, "aliases": [SERVED], "tags": [], "object": "model",
                                       "created": 1, "owned_by": "llamacpp",
                                       "meta": {"n_ctx": 262144, "n_ctx_train": 262144}}]}
# two slots, a window each (-c 524288 -np 2): each slot's context is its own window (n_ctx_slot = 262144, no kv_unified)
PROPS = {"default_generation_settings": {"n_ctx": 262144}, "total_slots": 2, "model_alias": SERVED,
         "model_path": "/root/.cache/huggingface/hub/x.gguf", "build_info": "b12316-5d4c8eeb2"}
METRICS = ("# HELP llamacpp:tokens_predicted_total Number of generation tokens processed\n"
           "# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total 1.23457e+06\n"
           "# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter\nllamacpp:spec_decode_num_accepted_tokens_total 300\n"
           "# TYPE llamacpp:spec_decode_num_drafts_total counter\nllamacpp:spec_decode_num_drafts_total 100\n"
           "# TYPE llamacpp:requests_processing gauge\nllamacpp:requests_processing 1\n"
           "# TYPE llamacpp:requests_deferred gauge\nllamacpp:requests_deferred 2\n")
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


class Engine(http.server.BaseHTTPRequestHandler):
    """llama-server as the cockpit meets it: the key on every route but /health."""
    asked: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        out = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        path = self.path.split("?")[0]
        Engine.asked.append(path)
        if path == "/health":
            return self._send(200, '{"status":"ok"}')
        if self.headers.get("Authorization") != "Bearer test-key":
            return self._send(401, '{"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}}')
        if path == "/v1/models":
            self._send(200, json.dumps(MODELS))
        elif path == "/props":
            self._send(200, json.dumps(PROPS))
        elif path == "/metrics":
            self._send(200, METRICS, "text/plain; version=0.0.4")
        else:
            self._send(404, '{"error":{"code":404,"message":"File Not Found","type":"not_found_error"}}')


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = Path(tempfile.mkdtemp(prefix="cockpit-flash-unsloth-"))
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.box), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
                          COCKPIT_ENGINE=f"http://127.0.0.1:{cls.engine.server_port}",
                          # never the box's own proxy: a stop goes there first since v1.40.3
                          COCKPIT_PROXY="http://127.0.0.1:1")
        (cls.box / "api-key").write_text("test-key\n")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_flash_unsloth_under_test", DASH / "cockpit.py")
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)
        # its unit as its installer renders it, the files it names in this box's own cache
        cls.hf = cls.box / "hf"
        snap = cls.hf / "hub" / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / ("d" * 40) / "UD-Q4_K_XL"
        draft = cls.hf / "hub" / "models--ggml-org--Qwen3.8-Flash-Next-GGUF" / "snapshots" / ("b" * 40)
        cls.files = [snap / f"m-0000{i}-of-00004.gguf" for i in range(1, 5)] + [draft / "d.gguf"]
        cls.template = cls.box / "chat-template-flash-unsloth.jinja"
        cls.check = cls.box / "llamacpp-files.sh"
        in_box = "/root/.cache/huggingface/hub"
        cls.unit = cls.box / U
        cls.unit.write_text(
            f"ExecStart=/bin/bash -c 'check={cls.check}; [ -f \"$$check\" ] || exit 78; \\\n"
            f"  bash \"$$check\" {U} flash-unsloth {' '.join(map(str, cls.files))} {cls.template} || exit $$?; \\\n"
            "  exec /usr/bin/docker run --rm --name qwen38-flash-unsloth --gpus all \\\n"
            f"  -v {cls.hf}:/root/.cache/huggingface:ro \\\n"
            f"  --mount type=bind,source={cls.template},target=/out/chat-template-flash-unsloth.jinja,readonly \\\n"
            "  ghcr.io/ggml-org/llama.cpp@sha256:" + "0" * 64 + " \\\n"
            f"    -m {in_box}/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/{'d' * 40}/UD-Q4_K_XL/m-00001-of-00004.gguf \\\n"
            f"    --alias {SERVED} \\\n"
            "    --jinja --chat-template-file /out/chat-template-flash-unsloth.jinja \\\n"
            "    --load-mode none --lazy-mode on \\\n"
            "    -c 524288 -np 2  \\\n"
            f"    -md {in_box}/models--ggml-org--Qwen3.8-Flash-Next-GGUF/snapshots/{'b' * 40}/d.gguf \\\n"
            "    --spec-type ngram-mod,draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0.5 \\\n"
            "    --metrics --no-ui --api-key-file /out/api-key --host 127.0.0.1 --port 30000'\n")
        cls.ck.FLASH_UNSLOTH_UNIT_PATH = cls.unit
        for name in ("NEMOTRON_UNIT_PATH", "NEMOTRON_SUPER_UNIT_PATH", "ORNITH_UNIT_PATH", "FLASH_LOCAL_UNIT_PATH"):
            setattr(cls.ck, name, cls.box / f"absent-{name.lower()}.service")
        cls.real_run = cls.ck.run

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls.engine.server_close()
        shutil.rmtree(cls.box, ignore_errors=True)

    def setUp(self):
        self.ck.run = self.real_run

    def serving(self, state="ready", unit=U):
        with self.ck.LIFE_LOCK:
            self.ck.LIFE["states"] = {unit: state} if state else {}

    def whole(self):
        """Every file its unit names there, and the check of them."""
        for f in self.files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"GGUF")
        self.template.write_text("{{ messages }}")
        self.check.write_text("#!/bin/bash\nexit 0\n")


class ItsEnginePanel(Base):
    def test_what_it_serves_comes_from_the_engine_with_the_key_and_from_its_own_unit(self):
        self.ck.run = lambda argv, **kw: ("Environment=PROMPT_CEILING_TOKENS=0 FLASH_PROMPT_CEILING_TOKENS=250000\n"
                                          if argv[:3] == ["systemctl", "show", "qwen38-keepalive.service"] else "")
        self.ck.UNIT_TARGET_CACHE.clear()
        out = self.ck.llamacpp_engine_info()
        info = out["info"]
        self.assertEqual((info["served_model_name"], info["model_path"]), (SERVED, REPO_ID))
        # one request's window from the engine, the windows of its two requests from its unit's -c
        self.assertEqual((info["context_length"], info["max_total_num_tokens"], info["max_running_requests"]),
                         (262144, 524288, 2))
        # llama.cpp's own KV cache type when the unit names none, Unsloth's dynamic 4-bit, its draft and its length
        self.assertEqual((info["kv_cache_dtype"], info["quantization"]), ("f16", "ud-q4_k_xl"))
        self.assertEqual((info["speculative_algorithm"], info["speculative_num_draft_tokens"]), ("n-grams + DFlash2", 7))
        # the version /props says (its build_info), in the form dashboard/engines.py reports for the pin
        self.assertEqual(info["version"], PROPS["build_info"])
        self.assertEqual(self.ck.en.ENGINES[U]["reports"], PROPS["build_info"])
        self.assertEqual(out["served_target"], "flash-unsloth")
        # the flash lane's one-prompt ceiling, which the proxy applies to every name of the flash model
        self.assertEqual(out["prompt_ceiling_tokens"], 250000)

    def test_its_load_is_read_off_its_metrics_with_the_key(self):
        self.serving("ready")
        load = self.ck.llamacpp_load(timeout=3)[0]
        self.assertEqual((load["num_reqs"], load["num_waiting_reqs"]), (3, 2))
        self.assertAlmostEqual(load["gen_tokens"], 1_234_570)
        # no pool and no prompt progress: llama-server publishes neither
        self.assertEqual((load["num_tokens"], load["prefix_queries"], load["finished"]), (0, None, None))


class WhatItDraftsWith(Base):
    def test_its_unit_says_it(self):
        # the n-gram lookup before the DFlash2 draft (2026-11-12), the draft alone, the lookup alone, nothing
        for spec, said in (("ngram-mod,draft-dflash", "n-grams + DFlash2"), ("draft-dflash", "DFlash2"),
                           ("ngram-mod", "n-grams"), ("none", None)):
            with self.subTest(spec=spec):
                self.assertEqual(self.ck.llamacpp_spec_algorithm(U, f"x --spec-type {spec} --spec-draft-n-max 7 y"), said)
        self.assertIsNone(self.ck.llamacpp_spec_algorithm(U, "x -c 262144 y"))


class ItsWindow(Base):
    def test_one_requests_window_whether_its_requests_share_one_or_have_one_each(self):
        # -c shared by the requests (-kvu), or holding a window for each without it (FLASH_UNSLOTH_SHARED, 2026-11-12):
        # the engine table says one request's, as lanes.sh's llamacpp_window does for opencode
        for text, want in (("x -c 262144 -np 4 -kvu  \\\n    -md y", 262144), ("x -c 524288 -np 2  \\\n    -md y", 262144),
                           ("x -c 1048576 -np 4 -ctk q8_0 -ctv q8_0 \\\n", 262144), ("x -c 262144 -np 1 \\\n", 262144),
                           ("#   -c 999 -np 9, a comment\nx -c 262144 -np 4 -kvu", 262144), ("x --alias y", None),
                           # its 1M mode, one request at a time (FLASH_UNSLOTH_CONTEXT=1m, 2026-11-12)
                           ("x -c 1048576 -np 1 -ctk q8_0 -ctv q8_0 --rope-scaling yarn \\\n    -md y", 1048576)):
            with self.subTest(text=text):
                self.assertEqual(self.ck.llamacpp_window(text), want)


class ItsState(Base):
    def setUp(self):
        super().setUp()
        self.asked = []
        self.ck.run = lambda argv, **kw: "abc\n" if argv[:2] == ["docker", "ps"] and "name=^qwen38-flash-unsloth$" in argv else ""

    def state(self, healthy, enter="100", prev="ready", log=()):
        self.ck.kolibri_healthy = lambda: healthy
        self.ck._llamacpp_log = lambda container: (self.asked.append(container), list(log))[1]
        st, boot, running = self.ck.llamacpp_engine_state(U, active="active", sub="running",
                                                         prev_state=prev, enter_key=enter)
        return st["state"], boot, running

    def test_a_boot_is_read_from_its_own_containers_log_with_its_own_reader(self):
        upto = next(i for i, ln in enumerate(BOOT) if "loading draft model" in ln) + 1
        st, boot, running = self.state(False, prev="starting", log=BOOT[:upto])
        self.assertEqual(st, "loading-draft")
        self.assertEqual(boot["detail"], "the DFlash2 draft")
        self.assertTrue(running)
        self.assertEqual(self.asked, ["qwen38-flash-unsloth"])

    def test_healthy_it_is_ready_and_its_log_is_not_read(self):
        self.assertEqual(self.state(True, prev="warming-up")[0], "ready")
        self.assertEqual(self.asked, [])


class WhatItDoesNotHave(Base):
    def test_the_canary_stands_down_and_says_why_in_its_name(self):
        self.serving("ready")
        out = self.ck.collect_canary()
        self.assertTrue(out["skipped"], out)
        self.assertIn("the Qwen3.8 Flash-Next Unsloth GGUF lane is checked by llama.cpp's own /health", out["why"])

    def test_flush_abort_and_fit_are_refused_in_its_name(self):
        self.serving("ready")
        for name in ("flush_cache", "abort_all", "fit_opencode"):
            with self.subTest(name=name):
                why = self.ck.llamacpp_cannot(name)
                self.assertTrue(why and why.startswith("The Qwen3.8 Flash-Next Unsloth GGUF lane"), why)
        with self.assertRaises(RuntimeError) as e:
            self.ck.engine_abort_all()
        self.assertIn("the Qwen3.8 Flash-Next Unsloth GGUF lane's llama.cpp has no route that aborts", str(e.exception))
        ok, lines, _ = self.ck.job_flush_cache(None)
        self.assertFalse(ok)
        self.assertIn("the Qwen3.8 Flash-Next Unsloth GGUF lane's llama.cpp has no cache flush route", lines[0])
        # and on no llama.cpp lane, none of its words
        self.serving("ready", unit="qwen38-sglang.service")
        self.assertIsNone(self.ck.llamacpp_cannot("flush_cache"))

    def test_system_one_points_at_the_sglang_lanes(self):
        self.serving("ready")
        out = self.ck.systemone_available(max_age=0)
        self.assertFalse(out["available"])
        self.assertIn("the Qwen3.8 Flash-Next Unsloth GGUF lane is serving", out["reason"])


class ItsUnit(Base):
    def test_the_target_is_read_from_the_path_of_its_model(self):
        self.ck.UNIT_TARGET_CACHE.clear()
        self.assertEqual(self.ck.unit_target(U), {"model": REPO_ID, "target": "flash-unsloth"})

    def test_it_is_a_switch_target_a_unit_the_page_may_drive_and_a_lane_it_prepares(self):
        self.assertIn("flash-unsloth", self.ck.ACTIONS["switch"]["params"]["target"])
        self.assertIn(U, self.ck.SERVING_UNITS)
        self.assertIn(U, self.ck.lc.ENGINE_UNITS)
        self.assertNotIn(U, self.ck.lc.TEXT_UNITS, "the SGLang belts are not its")
        self.assertEqual(self.ck.TARGET_PIN["flash-unsloth"], "FLASH_UNSLOTH_REV")
        self.assertEqual(self.ck.rg.PIN_MODELS["FLASH_UNSLOTH_REV"], REPO_ID)
        self.assertEqual(self.ck.lc.LANE_PROVIDER[U], "flash-unsloth")
        self.assertEqual(self.ck.ACTIONS["prepare"]["argv"]({"target": "flash-unsloth"})[-1], "flash-unsloth")

    def test_the_flash_lanes_prefix_does_not_take_it(self):
        self.assertEqual(self.ck.target_lane("flash-unsloth"), "flash-unsloth")
        self.assertEqual(self.ck.target_unit("flash-unsloth"), self.ck.FLASH_UNSLOTH_UNIT_PATH)

    def test_it_is_prepared_once_the_image_its_unit_names_is_here(self):
        img = "ghcr.io/ggml-org/llama.cpp@sha256:" + "0" * 64
        self.ck.run = lambda argv, **kw: "sha256:abc\n" if argv[:3] == ["docker", "image", "inspect"] and argv[-1] == img else ""
        self.assertTrue(self.ck.lane_prepared("flash-unsloth"))
        self.ck.run = lambda argv, **kw: ""
        self.assertFalse(self.ck.lane_prepared("flash-unsloth"))


class ItStartsOnlyWithItsFilesThere(Base):
    """A Start with no switch runs the check of its files, then llama-server on them: a part gone, an empty one or the
    check itself gone, and its unit refuses every start (exit 78), so the page does not offer it."""

    def setUp(self):
        super().setUp()
        img = "ghcr.io/ggml-org/llama.cpp@sha256:" + "0" * 64
        self.ck.run = lambda argv, **kw: "sha256:abc\n" if argv[:3] == ["docker", "image", "inspect"] and argv[-1] == img else ""
        self.whole()

    def test_whole_it_can_start(self):
        self.assertTrue(self.ck.lane_startable("flash-unsloth"))

    def test_a_part_gone_empty_or_its_check_gone_holds_the_start_back(self):
        for what, undo, redo in (
                ("a part gone", lambda: self.files[2].unlink(), self.whole),
                ("the draft empty", lambda: self.files[4].write_bytes(b""), self.whole),
                ("the template empty", lambda: self.template.write_text(""), self.whole),
                ("the check gone", lambda: self.check.unlink(), self.whole)):
            with self.subTest(what):
                undo()
                try:
                    self.assertFalse(self.ck.lane_startable("flash-unsloth"))
                finally:
                    redo()
        self.assertTrue(self.ck.lane_startable("flash-unsloth"))


if __name__ == "__main__":
    unittest.main()
