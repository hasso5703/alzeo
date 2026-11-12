#!/usr/bin/env python3
"""A llama.cpp lane's GGUF files, made whole in the Hugging Face cache with Python's standard library alone.

ckpt-fetch.py is the fetch of every other lane's checkpoint, with huggingface_hub, inside the lane's own
image or venv. llama.cpp's server image has no Python (the server stage of its .devops/cuda.Dockerfile
is a CUDA runtime, curl and llama-server: read at b12291, 2026-11-12), and the host has no
huggingface_hub. A llama.cpp lane serves a few files of a GGUF repository that holds many (every
quantization of the model, hundreds of GB), each pinned by its sha256 and its size, the way the speech
lane pins its GGUF (install-speech.sh). This fetches those files and nothing else, at the pinned
revision, into the cache's own layout: blobs/<sha256>, the name huggingface_hub gives a file it keeps
in LFS, and snapshots/<revision>/<path>, a relative link to it. A file huggingface_hub fetched before
is the same file here, with not one byte downloaded again, and one fetched here is the same for it.

    gguf-fetch.py HF_HOME REPO REVISION PATH=SHA256:SIZE [PATH=SHA256:SIZE ...]

A file whose blob is in the cache with its size answers without the network (a Load on a box that
holds it, offline or throttled): its content is checked when it is downloaded, as huggingface_hub
trusts a blob it wrote. One that is not is fetched from https://huggingface.co (HF_ENDPOINT moves it)
into blobs/<sha256>.incomplete, which a later run resumes, then checked against its sha256 and its
size before it takes its name: a file that does not match is never kept. Refused before the first byte
when the cache's disk lacks the room the missing bytes need. Five attempts per file, each resuming.
The token, HF_TOKEN, is read from the environment, never from the command line, and goes to the Hub
only: a header that is not passed on when the Hub redirects to its CDN. Each blob is fetched under the
lock huggingface_hub takes for it (.locks/<repo folder>/<sha256>.lock), so the two never write one
blob at once. The byte count is hf-progress.py's, in its words, when it is beside this file. The
snapshot's folder is the last line printed.
"""
import argparse
import contextlib
import fcntl
import hashlib
import http.client
import importlib.util
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ATTEMPTS = 5
CHUNK = 8 << 20
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


class Refused(SystemExit):
    """What makes a fetch stop for good, said in one line: no retry changes it."""


def spec(text):
    """PATH=SHA256:SIZE -> (path, sha256, size), each checked: a path inside the snapshot, a full digest."""
    path, sep, rest = text.rpartition("=")
    sha, _, size = rest.partition(":")
    parts = path.split("/")
    if not sep or not path or path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise argparse.ArgumentTypeError(f"{text}: not PATH=SHA256:SIZE with a path inside the repository")
    if not SHA.fullmatch(sha) or not size.isdigit() or int(size) <= 0:
        raise argparse.ArgumentTypeError(f"{text}: the sha256 is 64 lowercase hex digits and the size a number of bytes")
    return path, sha, int(size)


def repo_dir(hf_home, repo):
    return os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))


def whole(blob, size):
    """The blob is in the cache with its size (a file appears under its sha256 only once whole)."""
    try:
        return os.path.isfile(blob) and os.path.getsize(blob) == size
    except OSError:
        return False


def link(repo, rev, path, sha):
    """snapshots/<rev>/<path> -> the blob, relative as huggingface_hub writes it, replaced in one step."""
    dest = os.path.join(repo, "snapshots", rev, *path.split("/"))
    target = os.path.relpath(os.path.join(repo, "blobs", sha), os.path.dirname(dest))
    if os.path.islink(dest) and os.readlink(dest) == target:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.link"
    with contextlib.suppress(FileNotFoundError):
        os.unlink(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, dest)
    return dest


def free_bytes(path):
    """The room on the disk of a path, or of its nearest folder that exists."""
    while not os.path.exists(path):
        path = os.path.dirname(path) or "/"
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def partial(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def url_of(repo, rev, path):
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"


class _TokenOnTheSameHost(urllib.request.HTTPRedirectHandler):
    """The token goes on with a redirect that stays on the host it was sent to (a repository renamed, a Location of
    the Hub's own), never to another host (the CDN the Hub sends a file to). urllib drops a header added with
    add_unredirected_header on every redirect, the Hub's own included, and a gated file moved on the Hub was then
    refused for want of a token that was set (code review of v1.48.0, 2026-11-12)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        token = req.unredirected_hdrs.get("Authorization")
        if new is not None and token:
            was, now = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(new.full_url)
            if (was.scheme, was.hostname, was.port) == (now.scheme, now.hostname, now.port):
                new.add_unredirected_header("Authorization", token)
        return new


OPENER = urllib.request.build_opener(_TokenOnTheSameHost)


def download(url, part, size, timeout):
    """The bytes of url into part, from where part ends: a server that answers the range whole (200)
    starts it over, one that says it is past the end (416) leaves it as it is."""
    have = partial(part)
    if have > size:
        os.unlink(part)
        have = 0
    if have == size:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "alzeo-gguf-fetch"})
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        # not passed on to the CDN the Hub redirects to, kept on the Hub's own redirects (_TokenOnTheSameHost)
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:
            return
        if e.code in (401, 403):
            raise Refused(f"the Hub refused {url} (HTTP {e.code}): a gated repository needs HF_TOKEN=<your token>") from None
        if e.code == 404:
            raise Refused(f"{url} is not on the Hub (HTTP 404): the pinned revision or file is gone upstream") from None
        raise
    with resp:
        mode = "ab" if have and resp.status == 206 else "wb"
        with open(part, mode) as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                if f.tell() > size:
                    break


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


@contextlib.contextmanager
def blob_lock(hf_home, repo, sha):
    """huggingface_hub's own lock of this blob (filelock, an flock on the same file)."""
    folder = os.path.join(hf_home, "hub", ".locks", "models--" + repo.replace("/", "--"))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{sha}.lock"), "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def fetch_one(hf_home, repo, rev, path, sha, size, timeout, say):
    folder = repo_dir(hf_home, repo)
    blob = os.path.join(folder, "blobs", sha)
    with blob_lock(hf_home, repo, sha):
        if whole(blob, size):
            return
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        part = blob + ".incomplete"
        url = url_of(repo, rev, path)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                download(url, part, size, timeout)
                if partial(part) != size:
                    raise OSError(f"{path}: {partial(part)} bytes of {size}")
                break
            except Refused:
                raise
            # a connection cut, a timeout, a 5xx or a 429 (HTTPError is an OSError), a body cut short
            except (OSError, http.client.HTTPException) as e:
                if attempt == ATTEMPTS:
                    raise Refused(f"{path}: the download failed {ATTEMPTS} times ({type(e).__name__}: {e}); "
                                  "run again to resume, and set HF_TOKEN if the Hub throttles it") from None
                say(f"download interrupted ({type(e).__name__}), resuming ({attempt}/{ATTEMPTS})...")
                time.sleep(10)
        got = digest(part)
        if got != sha:
            os.unlink(part)
            raise Refused(f"{path} does not match its pinned sha256 (got {got[:12]}, pinned {sha[:12]}): not kept")
        os.chmod(part, 0o644)
        os.replace(part, blob)


def reporting(blob_dir, sizes, say):
    """hf-progress.py's byte count of the files to fetch, from beside this file; nothing when it is not there."""
    try:
        spec_ = importlib.util.spec_from_file_location(
            "hf_progress", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf-progress.py"))
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 (a progress line is never worth a download)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def watching():
        stop = threading.Event()
        thread = threading.Thread(target=mod._watch, args=(blob_dir, sizes, 0, 10.0, stop, say), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)
    return watching()


def say_line(line):
    print(line, flush=True)


def main(argv=None, say=say_line):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf_home", help="the Hugging Face home: its hub/ holds the cache")
    ap.add_argument("repo")
    ap.add_argument("rev", help="the pinned commit, 40 hex digits")
    ap.add_argument("files", nargs="+", type=spec, metavar="PATH=SHA256:SIZE")
    a = ap.parse_args(argv)
    if not REPO.fullmatch(a.repo):
        raise Refused(f"{a.repo}: not a repository id (owner/name)")
    if not COMMIT.fullmatch(a.rev):
        raise Refused(f"{a.rev}: a pinned revision is a full commit, 40 hex digits")
    timeout = float(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") or 30)
    folder = repo_dir(a.hf_home, a.repo)
    blob_dir = os.path.join(folder, "blobs")
    todo = [(p, s, n) for p, s, n in a.files if not whole(os.path.join(blob_dir, s), n)]
    if todo:
        if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
            raise Refused(f"HF_HUB_OFFLINE is set and {len(todo)} file(s) of {a.repo} are not in the cache: {todo[0][0]} first")
        sizes = {s: n for _, s, n in todo}
        need = sum(n - min(partial(os.path.join(blob_dir, s + ".incomplete")), n) for s, n in sizes.items())
        free = free_bytes(blob_dir)
        if free < need:
            raise Refused(f"{free / 2**30:.0f} GiB free in the cache, and {a.repo} needs about "
                          f"{-(-need // 10**9)} GB more: free some space first")
        with reporting(blob_dir, sizes, say):
            for p, s, n in todo:
                fetch_one(a.hf_home, a.repo, a.rev, p, s, n, timeout, say)
    for p, s, n in a.files:
        if not whole(os.path.join(blob_dir, s), n):
            raise Refused(f"{p}: not whole in the cache after its fetch")
        link(folder, a.rev, p, s)
    say(os.path.join(folder, "snapshots", a.rev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
