#!/usr/bin/env python3
"""The Qwen3.8 Flash-Next Unsloth GGUF lane (install-flash-unsloth.sh, v1.48), the first on llama.cpp: the unit its
--stage writes for a switch, the template it patches out of the GGUF, the fetch of its pinned GGUF files
(gguf-fetch.py), the check of its files at its start (llamacpp-files.sh), and what the switch, the pins, the proxy
and the uninstall say of it.

--stage runs the installer as written, in a throwaway home, with no sudo and no docker, its GGUF's first part a
small GGUF this test writes (its header alone: the chat template is all the stage reads of it). The fetch runs
against a server on a loopback port of the kernel's choosing, never the Hub."""
import hashlib
import http.server
import importlib.util
import io
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-flash-unsloth.sh").read_text()
PINS = dict(re.findall(r'^(FLASH_UNSLOTH_[A-Z0-9_]+)="([^"]*)"', TEXT, re.M))
PART1_PATH = PINS["FLASH_UNSLOTH_PART1"].rsplit("=", 1)[0]
DRAFT_PATH = PINS["FLASH_UNSLOTH_DRAFT"].rsplit("=", 1)[0]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PT = load("patch_template_for_unsloth", REPO / "patch-template.py")
GF = load("gguf_fetch_under_test", REPO / "gguf-fetch.py")
# Unsloth's template as the patcher meets it: its five places, in Unsloth's words where it has its own
UNSLOTH_TEMPLATE = "\n".join(["HEAD", PT.EFFORT_ANCHOR_UNSLOTH, PT.MSG_ANCHOR, PT.LEAN_ANCHOR, PT.SYSTEM_ANCHOR_UNSLOTH,
                              PT.TOOL_REF_ANCHOR, "TAIL"])


_PROXY = {}


def proxy():
    """keepalive-proxy.py as a module, once, in a throwaway home with no engine and no pair (as test_proxy_pair.py
    loads it): its import reads its argv and its environment."""
    if "m" not in _PROXY:
        tmp = tempfile.mkdtemp(prefix="unsloth-proxy-")
        saved, saved_argv = dict(os.environ), sys.argv
        os.environ.update({"HOME": tmp, "QWEN38_PAIR_FILE": os.path.join(tmp, "pair.json"),
                           "UPSTREAM": "http://127.0.0.1:9", "PROXY_BIND": "127.0.0.1"})
        sys.argv = ["keepalive-proxy.py"]
        err, sys.stderr = sys.stderr, io.StringIO()
        try:
            _PROXY["m"] = load(f"kp_for_unsloth_{time.time_ns()}", REPO / "keepalive-proxy.py")
        finally:
            sys.stderr, sys.argv = err, saved_argv
            os.environ.clear()
            os.environ.update(saved)
            shutil.rmtree(tmp, ignore_errors=True)
    return _PROXY["m"]


def gguf(template, extra_before=True):
    """A GGUF of version 3 with no tensor: a key before the template (as the real one has its tokenizer's), then
    tokenizer.chat_template."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = []
    if extra_before:
        kv.append(s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + s("a") + s("bc"))
        kv.append(s("general.sampling.top_k") + struct.pack("<I", 5) + struct.pack("<i", 20))
    kv.append(s("tokenizer.chat_template") + struct.pack("<I", 8) + s(template))
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv)


def exec_command(unit: str) -> str:
    """The ExecStart command as bash runs it: systemd joins the lines a backslash ends, and hands bash a $ for each $$."""
    line = unit[unit.index("ExecStart=/bin/bash -c '") + len("ExecStart=/bin/bash -c '"):]
    line = line[:line.index("'\nExecStop=")]
    return line.replace("\\\n", " ").replace("$$", "$")


class TheStagedUnit(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="unsloth-stage-"))
        self.home = self.t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.cfg / "api-key").write_text("test-key\n")
        self.units = self.t / "units"
        self.units.mkdir()
        self.hf = self.t / "hf"
        self.out = self.t / "stage.service"
        self.snap = self.hf / "hub" / f"models--{PINS['FLASH_UNSLOTH_REPO'].replace('/', '--')}" / "snapshots" / PINS["FLASH_UNSLOTH_REV"]
        (self.snap / "UD-Q4_K_XL").mkdir(parents=True)
        (self.snap / PART1_PATH).write_bytes(gguf(UNSLOTH_TEMPLATE))

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def stage(self, **env):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "HF_CACHE": str(self.hf),
             "LANES_UNIT_DIR": str(self.units), **env}
        r = subprocess.run(["bash", str(REPO / "install-flash-unsloth.sh"), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_the_unit_is_whole_and_serves_the_pinned_files_on_the_pinned_image(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        unit = self.out.read_text()
        self.assertNotRegex(unit, r"__[A-Z][A-Z0-9_]*__", "a placeholder left")
        in_box = "/root/.cache/huggingface/hub"
        self.assertIn(f"  {PINS['FLASH_UNSLOTH_IMAGE']} \\\n", unit)
        self.assertIn(f"-m {in_box}/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/{PINS['FLASH_UNSLOTH_REV']}/{PART1_PATH} ", unit)
        self.assertIn(f"-md {in_box}/models--ggml-org--Qwen3.8-Flash-Next-GGUF/snapshots/{PINS['FLASH_UNSLOTH_DRAFT_REV']}/"
                      f"{DRAFT_PATH} ", unit)
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface:ro", unit)
        self.assertIn("--alias qwen3.8-flash-next-unsloth", unit)
        # -c holds a window for each request, or the one they share (FLASH_UNSLOTH_SHARED)
        slots, ctx = int(PINS["FLASH_UNSLOTH_SLOTS"]), int(PINS["FLASH_UNSLOTH_CTX"])
        whole = ctx if PINS["FLASH_UNSLOTH_SHARED"] == "1" or slots == 1 else ctx * slots
        self.assertIn(f"-c {whole} -np {slots} ", unit)
        self.assertIn(f"--spec-type {PINS['FLASH_UNSLOTH_SPEC']} --spec-draft-n-max 7 --spec-draft-p-min 0.5 "
                      "--spec-draft-sampling probabilistic", unit)
        r = subprocess.run(["bash", "-n", "-c", exec_command(unit)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # the check of its files is the head of its start, and in place
        self.assertTrue((self.cfg / "llamacpp-files.sh").is_file())

    def test_the_n_gram_table_stays_on_the_disk(self):
        """--lazy-mode on keeps the 26.8 GiB table on the disk, whatever the load mode (b12291 maps a lazy tensor in every
        mode). Without it the GB10 (an integrated GPU, which llama.cpp's auto mode loads without mmap) loads the table
        with the weights, and the box's guard stopped the load at 5 GiB free (2026-11-11)."""
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        unit = self.out.read_text()
        self.assertIn(f"--load-mode {PINS['FLASH_UNSLOTH_LOAD']} --lazy-mode on \\\n", unit)
        self.assertIn("-ngl 999 -fa on --no-repack", unit)
        # nothing reads the files through before the engine: no preload, no copy (measured, 2026-11-12)
        self.assertNotRegex(exec_command(unit), r"\b(cat|dd|vmtouch|cp|rsync)\b")

    def test_the_engine_listens_on_this_box_alone_and_asks_the_key_from_a_file(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        unit = self.out.read_text()
        self.assertIn("--host 127.0.0.1 --port 30000'", unit)
        # the image says 0.0.0.0 and 8080: set again, the command line aside (the unit's comments say the image's)
        self.assertIn("-e LLAMA_ARG_HOST=127.0.0.1 -e LLAMA_ARG_PORT=30000", unit)
        self.assertNotIn("0.0.0.0", exec_command(unit))
        self.assertIn("--api-key-file /out/api-key", unit)
        self.assertIn(f"--mount type=bind,source={self.cfg}/api-key,target=/out/api-key,readonly", unit)
        self.assertNotRegex(unit, r"--api-key\s+(?!-file)", "the key on a command line")
        self.assertNotIn("test-key", unit)
        # its web interface is off, its counters on, its image's check of port 8080 off
        self.assertIn("--metrics --no-ui", unit)
        self.assertIn("--no-healthcheck", unit)

    def test_the_stage_writes_the_gguf_template_with_the_qwen_fixes(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        tpl = (self.cfg / "chat-template-flash-unsloth.jinja").read_text()
        self.assertIn("<system-reminder>", tpl)
        self.assertNotIn("System message must be at the beginning", tpl)
        self.assertIn("item.type == 'tool_reference'", tpl)
        self.assertIn("reasoning_effort|default('lean')", tpl)
        unit = self.out.read_text()
        self.assertIn(f"--mount type=bind,source={self.cfg}/chat-template-flash-unsloth.jinja,"
                      "target=/out/chat-template-flash-unsloth.jinja,readonly", unit)
        self.assertIn("--jinja --chat-template-file /out/chat-template-flash-unsloth.jinja", unit)

    def test_its_start_checks_every_file_the_engine_reads(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        cmd = exec_command(self.out.read_text())
        m = re.search(r'bash "\$check" qwen38-flash-unsloth\.service flash-unsloth (.+?) \|\| exit', cmd)
        self.assertIsNotNone(m, cmd)
        repo_dir = self.hf / "hub" / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / PINS["FLASH_UNSLOTH_REV"]
        draft_dir = self.hf / "hub" / "models--ggml-org--Qwen3.8-Flash-Next-GGUF" / "snapshots" / PINS["FLASH_UNSLOTH_DRAFT_REV"]
        want = [str(repo_dir / PINS[f"FLASH_UNSLOTH_PART{i}"].rsplit("=", 1)[0]) for i in range(1, 5)]
        want += [str(draft_dir / DRAFT_PATH), str(self.cfg / "chat-template-flash-unsloth.jinja")]
        self.assertEqual(m.group(1).split(), want)

    def test_the_proxy_reads_its_name_checkpoint_and_slots_in_its_command(self):
        """The pair reads which lane a box serves, its checkpoint and its requests at once in its engine's command
        (--served-model-name, --revision, --max-num-seqs): llama-server says them with --alias, the snapshot of its -m
        and -np. Its draft's snapshot (-md) is another checkpoint, which the pair never reads as the lane's."""
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        cmd = exec_command(self.out.read_text())
        kp = proxy()
        self.assertEqual(kp.names_in([cmd]), frozenset({"qwen3.8-flash-next-unsloth"}))
        self.assertEqual(kp.revision_in([cmd]), PINS["FLASH_UNSLOTH_REV"])
        self.assertEqual(kp.slots_in([cmd]), int(PINS["FLASH_UNSLOTH_SLOTS"]))
        self.assertIsNone(kp.revision_in([f"-md /x/snapshots/{PINS['FLASH_UNSLOTH_DRAFT_REV']}/d.gguf"]))
        self.assertEqual(kp.names_in(["docker run --network-alias x img --alias-y z"]), frozenset())

    def test_the_lane_serves_with_the_values_the_night_measured(self):
        """The defaults, measured on 2026-11-11 and 12: the weights copied into the GPU's memory (--load-mode none: 27.6
        tok/s where mmap gave 24.9 side by side, b12291), two requests at once with a window of 262,144 tokens each, so
        that none can make the other fail (63.2 tok/s in all with two, against 54.2 for one, 20 GiB available at ready,
        b12291), the trained window with no YaRN, llama.cpp's own KV cache type, and the n-gram lookup before the
        DFlash2 draft (on the pinned image b12316: 51.8 tok/s in code, 123.7 in a rewrite of code)."""
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        cmd = exec_command(self.out.read_text())
        self.assertIn("--load-mode none --lazy-mode on", cmd)
        self.assertRegex(cmd, r"-c 524288 -np 2\s+-md ")
        self.assertNotIn("-kvu", cmd.split())
        self.assertNotIn("--rope-scaling", cmd)
        self.assertNotRegex(cmd, r"(?<![\w-])-ct[kv](?![\w-])")
        self.assertIn("--spec-type ngram-mod,draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0.5 "
                      "--spec-draft-sampling probabilistic", cmd)

    def staged_with(self, **values):
        """The unit an installer whose measured variables hold these values stages, from a copy of the repository:
        (returncode, its output, the engine's command)."""
        repo = self.t / "repo"
        if not repo.is_dir():
            repo.mkdir()
            for f in [*REPO.glob("*.sh"), *REPO.glob("*.py"), REPO / "qwen38-flash-unsloth.service.template"]:
                shutil.copy(f, repo / f.name)
        text = TEXT
        for var, value in values.items():
            text, n = re.subn(rf'^FLASH_UNSLOTH_{var}="[^"]*"$', f'FLASH_UNSLOTH_{var}="{value}"', text, flags=re.M)
            self.assertEqual(n, 1, var)
        (repo / "install-flash-unsloth.sh").write_text(text)
        self.out.unlink(missing_ok=True)
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "HF_CACHE": str(self.hf), "LANES_UNIT_DIR": str(self.units)}
        r = subprocess.run(["bash", str(repo / "install-flash-unsloth.sh"), "--stage", str(self.out)], capture_output=True,
                           text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr, exec_command(self.out.read_text()) if self.out.exists() else ""

    def test_the_measured_options_render_as_the_installer_sets_them(self):
        """The window past the trained one takes YaRN's options, more requests at once share the window, and a value
        the engine would refuse stops the stage: the installer's variables of the night's measures (2026-11-12)."""
        rc, out, cmd = self.staged_with(YARN="2", CTX="524288", SLOTS="8", SHARED="1", KV="q8_0", LOAD="dio")
        self.assertEqual(rc, 0, out)
        self.assertIn("--load-mode dio --lazy-mode on", cmd)
        self.assertRegex(cmd, r"-c 524288 -np 8 -kvu -ctk q8_0 -ctv q8_0 --rope-scaling yarn --yarn-orig-ctx 262144 "
                              r"--rope-scale 2\s")
        # FP8 is refused for this model's architecture ("FP8 cache is not supported for architecture 'qwen4exp'",
        # 2026-11-12): the render refuses it before the engine does
        for var, bad in (("YARN", "1"), ("LOAD", "fast"), ("CTX", "1m"), ("SLOTS", "0"), ("KV", "f8_e4m3"), ("SHARED", "2")):
            with self.subTest(var=var, value=bad):
                rc, out, _ = self.staged_with(**{var: bad})
                self.assertNotEqual(rc, 0)
                self.assertIn(f"FLASH_UNSLOTH_{var} must be", out)
                self.assertFalse(self.out.exists(), "a unit written all the same")

    def test_the_requests_share_one_window_or_have_one_each(self):
        """SHARED=1: one window of CTX the requests share (-kvu), which one may take whole, and requests that together
        pass it all fail at once (llama.cpp's server-context.cpp). SHARED=0: a window of CTX for each, -c holding them
        all and no -kvu, so that no request can make another fail. Either way one request's window is CTX, which
        lanes.sh's llamacpp_window reads in the unit for opencode's limits (the cockpit's, in its own test)."""
        for shared, slots, kv, c, rest in (("1", "4", "", 262144, "-np 4 -kvu"), ("0", "2", "", 524288, "-np 2"),
                                           ("0", "4", "q8_0", 1048576, "-np 4 -ctk q8_0 -ctv q8_0")):
            with self.subTest(shared=shared, slots=slots):
                rc, out, cmd = self.staged_with(SHARED=shared, SLOTS=slots, KV=kv)
                self.assertEqual(rc, 0, out)
                self.assertRegex(cmd, rf"-c {c} {rest}\s+-md ")
                self.assertEqual("-kvu" in cmd.split(), shared == "1")
                said = subprocess.run(["bash", "-c", '. "$1"; llamacpp_window "$2"', "_", str(REPO / "lanes.sh"), str(self.out)],
                                      capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": str(self.home)})
                self.assertEqual(said.stdout.strip(), "262144", said.stderr)
        # the lane's own: a window each, for two requests (2026-11-12)
        self.assertEqual((PINS["FLASH_UNSLOTH_SHARED"], PINS["FLASH_UNSLOTH_SLOTS"]), ("0", "2"))

    def test_a_stage_with_no_gguf_in_the_cache_writes_no_unit(self):
        (self.snap / PART1_PATH).unlink()
        rc, out = self.stage()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("the chat template could not be patched", out)
        self.assertFalse(self.out.exists(), "a unit naming a template that is not there")


class ThePatcherReadsAGguf(unittest.TestCase):
    def test_the_template_is_read_past_the_keys_before_it(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d, "x.gguf")
            f.write_bytes(gguf("{{ messages }}"))
            self.assertEqual(PT.gguf_template(str(f)), "{{ messages }}")

    def test_unsloths_words_get_the_fixes_qwens_get(self):
        with tempfile.TemporaryDirectory() as d:
            f, out = pathlib.Path(d, "x.gguf"), pathlib.Path(d, "out.jinja")
            f.write_bytes(gguf(UNSLOTH_TEMPLATE))
            r = subprocess.run([sys.executable, str(REPO / "patch-template.py"), f"--gguf={f}", str(out)],
                               capture_output=True, text=True, env={**os.environ, "LEAN_DEFAULT": "1"})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(r.stdout.count("applied"), 5, r.stdout)
            tpl = out.read_text()
            self.assertIn(PT.SYSTEM_PATCHED_UNSLOTH, tpl)
            self.assertNotIn(PT.EFFORT_ANCHOR_UNSLOTH, tpl)

    def test_a_file_that_is_no_gguf_or_has_no_template_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for name, body in (("no.gguf", b"NOPE" + b"\0" * 32), ("bare.gguf", b"GGUF" + struct.pack("<IQQ", 3, 0, 0))):
                f = pathlib.Path(d, name)
                f.write_bytes(body)
                r = subprocess.run([sys.executable, str(REPO / "patch-template.py"), f"--gguf={f}", str(pathlib.Path(d, "o"))],
                                   capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0, name)
                self.assertFalse(pathlib.Path(d, "o").exists(), name)


class Hub(http.server.ThreadingHTTPServer):
    """A Hub of a few files, at /<repo>/resolve/<rev>/<path>: ranges answered 206, every request written down with
    its headers; `redirect` sends a file's request to another server, as the Hub sends it to its CDN."""

    def __init__(self, files, redirect=None):
        self.files, self.redirect, self.seen = files, redirect, []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(h):
                h.server.seen.append((h.path, dict(h.headers)))
                rel = h.path.split("/resolve/", 1)[-1].split("/", 1)[-1]
                if h.server.redirect:
                    h.send_response(302)
                    h.send_header("Location", h.server.redirect + h.path)
                    h.end_headers()
                    return
                body = h.server.files.get(rel)
                if body is None:
                    h.send_response(404)
                    h.end_headers()
                    return
                start = 0
                rng = h.headers.get("Range")
                if rng:
                    start = int(rng.split("=", 1)[1].split("-", 1)[0])
                h.send_response(206 if rng else 200)
                h.send_header("Content-Length", str(len(body) - start))
                h.end_headers()
                h.wfile.write(body[start:])
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class TheFetch(unittest.TestCase):
    """gguf-fetch.py: the pinned files, into the cache's own layout, checked, resumed, offline when there."""

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="gguf-fetch-"))
        self.data = {"Q/model-00001-of-00002.gguf": b"first part " * 1000, "Q/model-00002-of-00002.gguf": b"second " * 3000}
        self.specs = [f"{p}={hashlib.sha256(b).hexdigest()}:{len(b)}" for p, b in self.data.items()]
        self.rev = "a" * 40
        self.saved = {k: os.environ.get(k) for k in ("HF_ENDPOINT", "HF_TOKEN", "HF_HUB_OFFLINE")}
        self.hubs = []

    def tearDown(self):
        for h in self.hubs:
            h.shutdown()
            h.server_close()
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(self.t, ignore_errors=True)

    def hub(self, files=None, redirect=None):
        h = Hub(self.data if files is None else files, redirect)
        self.hubs.append(h)
        return h

    def run_fetch(self, specs=None):
        out = io.StringIO()
        with redirect_stdout(out):
            GF.main([str(self.t), "org/model", self.rev, *(specs or self.specs)], say=print)
        return out.getvalue()

    def test_the_files_land_where_huggingface_hub_puts_them(self):
        os.environ["HF_ENDPOINT"] = self.hub().url
        out = self.run_fetch()
        snap = self.t / "hub" / "models--org--model" / "snapshots" / self.rev
        self.assertEqual(out.strip().splitlines()[-1], str(snap))
        for path, body in self.data.items():
            link = snap / path
            self.assertTrue(link.is_symlink(), path)
            self.assertEqual(os.readlink(link), f"../../../blobs/{hashlib.sha256(body).hexdigest()}")
            self.assertEqual(link.read_bytes(), body)
        self.assertEqual(sorted(p.name for p in (self.t / "hub" / "models--org--model" / "blobs").iterdir()),
                         sorted(hashlib.sha256(b).hexdigest() for b in self.data.values()))

    def test_files_already_whole_need_no_network(self):
        os.environ["HF_ENDPOINT"] = self.hub().url
        self.run_fetch()
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"      # nothing listens there
        self.run_fetch()
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.run_fetch()

    def test_a_cut_download_is_resumed_from_where_it_stopped(self):
        hub = self.hub()
        os.environ["HF_ENDPOINT"] = hub.url
        path, body = next(iter(self.data.items()))
        blobs = self.t / "hub" / "models--org--model" / "blobs"
        blobs.mkdir(parents=True)
        (blobs / (hashlib.sha256(body).hexdigest() + ".incomplete")).write_bytes(body[:100])
        self.run_fetch([s for s in self.specs if s.startswith(path)])
        self.assertEqual((blobs / hashlib.sha256(body).hexdigest()).read_bytes(), body)
        self.assertIn("bytes=100-", [h.get("Range") for _, h in hub.seen])

    def test_a_file_that_is_not_the_pinned_one_is_never_kept(self):
        path, body = next(iter(self.data.items()))
        os.environ["HF_ENDPOINT"] = self.hub({path: body + b"tampered"}).url
        spec = f"{path}={hashlib.sha256(body).hexdigest()}:{len(body) + 8}"
        with self.assertRaises(SystemExit) as e:
            self.run_fetch([spec])
        self.assertIn("does not match its pinned sha256", str(e.exception.code))
        blobs = self.t / "hub" / "models--org--model" / "blobs"
        self.assertEqual([p.name for p in blobs.iterdir() if not p.name.endswith(".lock")], [])

    def test_the_token_goes_to_the_hub_and_never_past_its_redirect(self):
        cdn = self.hub()
        hub = self.hub(files={}, redirect=cdn.url)
        os.environ["HF_ENDPOINT"] = hub.url
        os.environ["HF_TOKEN"] = "hf_secretvalue"
        self.run_fetch()
        self.assertTrue(hub.seen and all(h.get("Authorization") == "Bearer hf_secretvalue" for _, h in hub.seen))
        self.assertTrue(cdn.seen and all("Authorization" not in h for _, h in cdn.seen), cdn.seen)

    def test_offline_with_a_file_missing_says_so_and_fetches_nothing(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        with self.assertRaises(SystemExit) as e:
            self.run_fetch()
        self.assertIn("HF_HUB_OFFLINE", str(e.exception.code))

    def test_no_room_no_byte(self):
        hub = self.hub()
        os.environ["HF_ENDPOINT"] = hub.url
        saved, GF.free_bytes = GF.free_bytes, lambda path: 10
        try:
            with self.assertRaises(SystemExit) as e:
                self.run_fetch()
        finally:
            GF.free_bytes = saved
        self.assertIn("free some space first", str(e.exception.code))
        self.assertEqual(hub.seen, [])

    def test_a_pin_is_a_path_inside_the_repository_and_a_whole_digest(self):
        for bad in ("../x.gguf=" + "0" * 64 + ":1", "/x.gguf=" + "0" * 64 + ":1", "x.gguf=abc:1", "x.gguf=" + "0" * 64 + ":0"):
            with self.subTest(bad), self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                    GF.main([str(self.t), "org/model", self.rev, bad], say=print)


class TheStartCheck(unittest.TestCase):
    """llamacpp-files.sh: exit 78 with the line the cockpit reads when a file the engine reads is not whole."""

    def check(self, *files):
        r = subprocess.run(["bash", str(REPO / "llamacpp-files.sh"), "qwen38-flash-unsloth.service", "flash-unsloth", *files],
                           capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LLAMACPP_FILES_TRIES": "0"})
        return r.returncode, r.stderr

    def test_whole_files_start_the_engine_and_any_other_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            ok = pathlib.Path(d, "ok.gguf")
            ok.write_bytes(b"x")
            empty = pathlib.Path(d, "empty.gguf")
            empty.touch()
            dangling = pathlib.Path(d, "dangling.gguf")
            dangling.symlink_to(pathlib.Path(d, "no-blob"))
            self.assertEqual(self.check(str(ok))[0], 0)
            for bad in (empty, dangling, pathlib.Path(d, "absent.gguf")):
                with self.subTest(bad.name):
                    rc, err = self.check(str(ok), str(bad))
                    self.assertEqual(rc, 78)
                    self.assertIn(f"preflight: qwen38-flash-unsloth.service lacks its files: {bad} is missing", err)
                    self.assertIn("./switch-model.sh flash-unsloth", err)


class TheSwitchAndThePins(unittest.TestCase):
    SWITCH = (REPO / "switch-model.sh").read_text()

    def block(self):
        m = re.search(r'^if \[ "\$CHOICE" = "flash-unsloth" \]; then\n(.*?)^fi\n', self.SWITCH, re.M | re.S)
        self.assertIsNotNone(m)
        return m.group(1)

    def test_the_switch_fetches_through_the_installer_and_stops_there_on_a_fetch_alone(self):
        b = self.block()
        at = b.index('bash "$REPO_DIR/install-flash-unsloth.sh" --fetch')
        after = b[b.index("\n", b.index("|| die ", at)) + 1:].lstrip()
        self.assertTrue(after.startswith('fetched "$FLASH_UNSLOTH_REPO @'), after[:80])
        # and its stage after it, at the path the cockpit's sudoers names
        self.assertLess(at, b.index('bash "$REPO_DIR/install-flash-unsloth.sh" --stage "$FLU_STAGE"'))
        self.assertIn('sudo install -m 644 "$FLU_STAGE" "$FLASH_UNSLOTH_UNIT"', b)
        self.assertNotIn("ckpt-fetch.py", b)

    def test_the_installer_reads_its_pins_once(self):
        self.assertEqual(len(re.findall(r"^FLASH_UNSLOTH_PART[1-4]=", TEXT, re.M)), 4)
        for var in ("FLASH_UNSLOTH_PART1", "FLASH_UNSLOTH_PART2", "FLASH_UNSLOTH_PART3", "FLASH_UNSLOTH_PART4",
                    "FLASH_UNSLOTH_DRAFT"):
            self.assertRegex(PINS[var], r"^[^=]+\.gguf=[0-9a-f]{64}:\d+$", var)
        self.assertRegex(PINS["FLASH_UNSLOTH_IMAGE"], r"^ghcr\.io/ggml-org/llama\.cpp@sha256:[0-9a-f]{64}$")
        self.assertRegex(PINS["FLASH_UNSLOTH_REV"], r"^[0-9a-f]{40}$")
        self.assertRegex(PINS["FLASH_UNSLOTH_DRAFT_REV"], r"^[0-9a-f]{40}$")
        # the room it asks is the files' size, rounded up to the GB
        total = sum(int(PINS[v].rsplit(":", 1)[1]) for v in ("FLASH_UNSLOTH_PART1", "FLASH_UNSLOTH_PART2",
                                                              "FLASH_UNSLOTH_PART3", "FLASH_UNSLOTH_PART4", "FLASH_UNSLOTH_DRAFT"))
        self.assertEqual(int(PINS["FLASH_UNSLOTH_GB"]), -(-total // 10**9))

    def test_each_place_that_names_its_pins_names_these(self):
        pins = (REPO / "check-pins.sh").read_text()
        self.assertIn('check_model "unsloth-part$n" "$FLASH_UNSLOTH_REPO" "$FLASH_UNSLOTH_REV" "${part%=*}" "$sha"', pins)
        self.assertIn('check_image flash-unsloth  "$FLASH_UNSLOTH_IMAGE"', pins)
        engines = load("engines_for_unsloth", REPO / "dashboard" / "engines.py").ENGINES["qwen38-flash-unsloth.service"]
        self.assertEqual((engines["engine"], engines["pin"]), ("llama.cpp", PINS["FLASH_UNSLOTH_IMAGE"]))
        registry = load("registry_for_unsloth", REPO / "dashboard" / "registry.py")
        self.assertEqual(registry.PIN_MODELS["FLASH_UNSLOTH_REV"], PINS["FLASH_UNSLOTH_REPO"])
        self.assertEqual(registry.PIN_MODELS["FLASH_UNSLOTH_DRAFT_REV"], PINS["FLASH_UNSLOTH_DRAFT_REPO"])
        uninstall = (REPO / "uninstall.sh").read_text()
        for value in (PINS["FLASH_UNSLOTH_REPO"], PINS["FLASH_UNSLOTH_DRAFT_REPO"], PINS["FLASH_UNSLOTH_IMAGE"]):
            self.assertIn(f" {value}", uninstall)


class TheProxyKnowsLlamaCpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kp = proxy()

    def test_closing_is_its_abort_as_on_vllm(self):
        self.assertEqual(self.kp.CLOSE_ABORTS, frozenset({"vllm", "llamacpp"}))

    def test_its_window_is_the_context_its_models_route_says(self):
        rows = [{"id": "qwen3.8-flash-next-unsloth", "owned_by": "llamacpp", "meta": {"n_ctx": 262144}}]
        self.assertEqual(self.kp.served_window(rows), 262144)
        # a meta of another engine's is not read so
        self.assertEqual(self.kp.served_window([{"id": "x", "owned_by": "sglang", "meta": {"n_ctx": 5}}]), 0)

    def test_its_name_is_a_flash_name_and_a_lane_name(self):
        self.assertIn("qwen3.8-flash-next-unsloth", self.kp.FLASH_SERVED_NAMES)
        self.assertIn("qwen3.8-flash-next-unsloth", self.kp.LANE_NAMES)

    def test_system_one_says_where_it_serves(self):
        """System One reads its options' labels off top_logprobs as SGLang gives them, which nothing measured on
        llama.cpp: refused there, with the lanes it serves on named, as on vLLM."""
        saved = self.kp.served_models, self.kp._SERVED["engine"]
        self.kp.served_models = lambda: ("qwen3.8-flash-next-unsloth",)
        self.kp._SERVED["engine"] = "llamacpp"
        try:
            with self.assertRaises(self.kp.SystemOneUsage) as e:
                self.kp.systemone_model("jev-latest")
        finally:
            self.kp.served_models, self.kp._SERVED["engine"] = saved
        self.assertIn("qwen3.8-flash-next-unsloth on llama.cpp now", str(e.exception))


if __name__ == "__main__":
    unittest.main()
