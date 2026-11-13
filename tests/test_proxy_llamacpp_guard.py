"""The size guard on the llama.cpp lane counts what it judges, with the engine's own routes (code review of v1.48.0,
2026-11-12).

llama-server's /tokenize reads "content" alone and answers {"tokens": [...]} (server-context.cpp, b12291 and b12316):
asked with SGLang's shape, messages or a prompt, it answered an empty list, no body was ever counted on that lane, and
its size decided, a token a byte. Its pool was one slot's window, so a prompt got 92 % of that window and a wedge looked
possible: every chat body past 241,172 bytes was refused as too long, an opencode session near 60,000 real tokens. A
chat body is now counted by llama-server's /v1/chat/completions/input_tokens, a prompt by its /tokenize content, the
pool is its slots' windows together (/props total_slots), and the flash ceiling lifts past a window of 262,144 (YaRN),
as on the FP4-Local lane. A reasoning effort of "none" in chat_template_kwargs, which llama-server hands to the
template and the template refuses, becomes enable_thinking false, as on the other Qwen lanes.

The fake engine answers as llama-server does on those routes, and counts a word or a digit as one token.
"""
import http.server
import importlib.util
import json
import os
import re
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _keep_env(cls):
    """Put os.environ back as this class found it (tests/test_suite_isolation.py holds it)."""
    saved = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("kproxy_llamacpp_guard", HERE.parents[1] / "keepalive-proxy.py")
SERVED = "qwen3.8-flash-next-unsloth"


def tokens(text):
    return len(re.findall(r"\d|[^\d\s]+", text))


def text_of(messages):
    out = []
    for m in messages:
        c = m.get("content")
        out += [c] if isinstance(c, str) else [p.get("text", "") for p in c or () if isinstance(p, dict)]
    return " ".join(out)


class Engine(http.server.BaseHTTPRequestHandler):
    """llama-server as the proxy meets it: its models, props and metrics, its two count routes, and generations."""
    protocol_version = "HTTP/1.1"
    window, slots = 262144, 2
    generated, counted, auths, convs = [], [], [], []

    def log_message(self, *a):
        pass

    def _json(self, status, doc):
        out = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": SERVED, "object": "model", "owned_by": "llamacpp",
                                                         "meta": {"n_ctx": Engine.window, "n_ctx_train": 262144}}]})
        elif self.path == "/props":
            self._json(200, {"default_generation_settings": {"n_ctx": Engine.window}, "total_slots": Engine.slots})
        elif self.path == "/metrics":
            body = b"# TYPE llamacpp:requests_processing gauge\nllamacpp:requests_processing 0\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/tokenize":
            # its handler: "content" alone is read, anything else counts nothing (server-context.cpp:6458-6495)
            Engine.counted.append(self.path)
            content = body.get("content")
            n = tokens(content) if isinstance(content, str) else 0
            self._json(200, {"tokens": list(range(n))})
            return
        if self.path == "/v1/chat/completions/input_tokens":
            Engine.counted.append(self.path)
            self._json(200, {"input_tokens": tokens(text_of(body.get("messages") or [])),
                             "object": "response.input_tokens"})
            return
        auth = self.headers.get("Authorization")
        if auth != "Bearer test-key":
            # llama.cpp compares the "Bearer " prefix exactly (measured on the v1.48 lane,
            # 2026-11-12: a lowercase "bearer <key>", which the door admits as SGLang's own
            # rule does, came back 401 "Invalid API Key")
            self._json(401, {"error": {"code": 401, "message": "Invalid API Key",
                                       "type": "authentication_error"}})
            return
        Engine.auths.append(auth)
        Engine.convs.append(self.headers.get("X-Conversation-Id"))
        Engine.generated.append(self.path)
        self._json(200, {"id": "x", "object": "chat.completion", "model": SERVED,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                      "finish_reason": "stop"}]})


class LlamaCppGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS",
                  "PROMPT_CEILING_TOKENS", "FLASH_PROMPT_CEILING_TOKENS"):
            os.environ.pop(k, None)
        cls.mod = importlib.util.module_from_spec(SPEC)
        sys.argv = ["keepalive-proxy.py"]
        SPEC.loader.exec_module(cls.mod)
        cls.mod._api_key = lambda: "test-key"
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()

    def setUp(self):
        Engine.window, Engine.slots = 262144, 2
        Engine.generated.clear()
        Engine.counted.clear()
        Engine.auths.clear()
        Engine.convs.clear()
        self.mod.FLASH_PROMPT_CEILING_TOKENS = 0
        self.mod.invalidate_pool()

    def share(self, pool):
        return int(pool * (1.0 - self.mod.OVERSIZE_MARGIN_FRAC))

    def post(self, path, doc, auth="Bearer test-key", headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(doc).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": auth, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_its_pool_is_its_slots_windows_and_no_prompt_can_wedge_it(self):
        pool = self.mod.pool_tokens()
        self.assertEqual((pool, self.mod._POOL["window"]), (524288, 262144))
        # the engine refuses past its window itself: the average nominates, the count decides
        self.assertFalse(self.mod.wedge_possible(pool))
        self.assertEqual(self.mod.prompt_limit(pool), self.share(524288))

    def test_the_flash_ceiling_holds_at_its_own_window_and_lifts_past_it(self):
        self.mod.FLASH_PROMPT_CEILING_TOKENS = 250000
        self.assertEqual(self.mod.prompt_limit(self.mod.pool_tokens()), 250000)
        # one request at a time, a window of 1,048,576 by YaRN (2026-11-12): the pool's share, as on the FP4-Local lane
        Engine.window, Engine.slots = 1048576, 1
        self.mod.invalidate_pool()
        pool = self.mod.pool_tokens()
        self.assertEqual(pool, 1048576)
        self.assertEqual(self.mod.prompt_limit(pool), self.share(1048576))

    def test_an_agents_long_chat_is_relayed(self):
        # 480 kB of prose, 80,000 tokens, an opencode session's size: refused by its bytes before, a token a byte past
        # 241,172; now under the average's nomination, as on the other flash lanes
        doc = {"model": SERVED, "messages": [{"role": "user", "content": "lorem " * 80000}], "max_tokens": 8}
        status, body = self.post("/v1/chat/completions", doc)
        self.assertEqual(status, 200, body)
        self.assertEqual(Engine.generated, ["/v1/chat/completions"])

    def test_a_chat_the_average_nominates_is_counted_and_relayed_when_it_fits(self):
        # 1.5 MB, 250,000 tokens: nominated by its bytes, counted by the engine's chat count route, within the limit
        doc = {"model": SERVED, "messages": [{"role": "user", "content": "lorem " * 250000}], "max_tokens": 8}
        status, body = self.post("/v1/chat/completions", doc)
        self.assertEqual(status, 200, body)
        self.assertEqual(Engine.generated, ["/v1/chat/completions"])
        self.assertEqual(Engine.counted, ["/v1/chat/completions/input_tokens"])

    def test_a_chat_past_the_limit_is_refused_by_the_engines_count(self):
        doc = {"model": SERVED, "messages": [{"role": "user", "content": "lorem " * 500000}]}
        status, body = self.post("/v1/chat/completions", doc)
        self.assertEqual(status, 400, body)
        err = json.loads(body)["error"]
        self.assertEqual(err["code"], "context_length_exceeded")
        self.assertIn("500000 prompt tokens (counted by the engine)", err["message"])
        self.assertEqual(Engine.generated, [])

    def test_a_long_prompt_is_counted_by_its_tokenize_content(self):
        self.mod.pool_tokens()
        status, body = self.post("/v1/completions", {"model": SERVED, "prompt": "lorem " * 120000, "max_tokens": 8})
        self.assertEqual(status, 200, body)
        self.assertEqual(Engine.generated, ["/v1/completions"])
        self.assertEqual(self.mod.tokenize_count(json.dumps({"prompt": "a b c 1 2"}).encode(), "/v1/completions"), 5)

    def test_a_none_effort_in_the_templates_kwargs_turns_thinking_off(self):
        self.mod.pool_tokens()          # the engine named on the way: llama.cpp, serving the Unsloth GGUF lane
        for doc in ({"chat_template_kwargs": {"reasoning_effort": "none"}}, {"reasoning_effort": "none"}):
            with self.subTest(doc=doc):
                body = json.dumps({"model": SERVED, "messages": [{"role": "user", "content": "hi"}], **doc}).encode()
                out, changed = self.mod.qwen_effort_none(body, "/v1/chat/completions")
                self.assertTrue(changed)
                j = json.loads(out)
                self.assertEqual(j["chat_template_kwargs"], {"enable_thinking": False})
                self.assertNotIn("reasoning_effort", j)
        # a llama.cpp lane of another model keeps its body
        saved = self.mod._SERVED["names"]
        self.mod._SERVED["names"] = ("another-model",)
        try:
            body = json.dumps({"messages": [], "chat_template_kwargs": {"reasoning_effort": "none"}}).encode()
            self.assertEqual(self.mod.qwen_effort_none(body, "/v1/chat/completions"), (body, False))
        finally:
            self.mod._SERVED["names"] = saved

    def test_a_lowercase_bearer_scheme_reaches_the_engine_as_the_engine_spells_it(self):
        # the door reads the scheme in any case (its rule is SGLang's, measured 2026-10-08:
        # "bearer <key>" admitted), and v6.38 only stripped the whitespace around the value.
        # llama.cpp is stricter than SGLang there: the same well-formed key sent as
        # "bearer <key>" answered 401 on the live lane (2026-11-12), where every other lane
        # served it. The door has already compared the key byte for byte (engine_admits), so
        # the one thing the relay may still change is the scheme's spelling: name it the way
        # the engine spells it, which is what the door accepted.
        st, out = self.post("/v1/chat/completions",
                            {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]},
                            auth="bearer test-key")
        self.assertEqual(st, 200, out[:120])
        self.assertEqual(Engine.auths[-1], "Bearer test-key")
        # a wrong key under any spelling stays refused at the door, before the relay: the door's own body,
        # SGLang's, and not the engine's "Invalid API Key"
        st, out = self.post("/v1/chat/completions",
                            {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]},
                            auth="bearer wrong")
        self.assertEqual((st, out), (401, self.mod.ENGINE_401_BODY))
        self.assertEqual(Engine.generated, ["/v1/chat/completions"], "only the admitted one reached the engine")

    def test_a_conversation_id_never_reaches_the_engine(self):
        # llama-server attaches a request carrying X-Conversation-Id to a resumable stream, and then stops it only when
        # the stream is cancelled (DELETE /v1/stream), never when its connection closes: server_res_spipe::should_stop
        # and on_complete, tools/server/server-stream.cpp of b12291 (read 2026-11-13). Relayed, it would let a client
        # that left, and the memory floor's stop, leave a generation running to its end on one of the two windows.
        # The proxy does not use it, and SGLang and vLLM ignore it: it stays at the door, in any case.
        for name in ("X-Conversation-Id", "x-conversation-id"):
            with self.subTest(header=name):
                Engine.convs.clear()
                st, out = self.post("/v1/chat/completions",
                                    {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]},
                                    headers={name: "conv-1"})
                self.assertEqual(st, 200, out[:120])
                self.assertEqual(Engine.convs, [None])


if __name__ == "__main__":
    unittest.main()
