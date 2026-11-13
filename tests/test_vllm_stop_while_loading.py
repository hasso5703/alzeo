#!/usr/bin/env python3
"""A vLLM lane stopped while its engine loads ends stopped, not failed; a crash of its own stays one.

systemctl stop qwen38-flash-local 16 s into a start (the reference box, 2026-11-13): docker stop sent
the container SIGTERM, vLLM turned it into KeyboardInterrupt("terminated") while it initialized and,
PID 1 in its container, exited 1. docker run, which the unit's main process exec'ed, returned it, 1 is
not in SuccessExitStatus=137 143, and systemd said "Failed with result 'exit-code'": the lane's card
showed a failure for a plain stop. Every vLLM lane serves through the same entrypoint, which turns a
SIGTERM into that interrupt until uvicorn takes the signal over. A stop now marks itself in the unit's
runtime directory before it stops the engine, and the main process, which waits on docker run, exits
143 for a stop it was asked, whatever the engine's code, else with docker run's own.

These run each vLLM unit's ExecStart and ExecStop lines as systemd hands them to bash (a backslash at a
line's end joins it to the next, $$ is a $, a leading - ignores the command's failure), in order, with
a docker that stands in for the engine: its run loads until it is stopped or crashes, then exits 1 as
vLLM did, and its stop ends the run and waits for it, as docker stop does.
"""
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
VLLM_UNITS = ("qwen38-flash-local", "qwen38-kolibri", "qwen38-nemotron", "qwen38-nemotron-super",
              "qwen38-ornith-122b")

FAKE_DOCKER = """#!/bin/bash
state="$FAKE_DOCKER_STATE"
case "$1" in
  run)
    : > "$state/loading"
    for _ in $(seq 600); do [ -e "$state/end" ] && break; sleep 0.05; done
    rm -f "$state/loading"
    exit 1 ;;
  stop)
    : > "$state/end"
    for _ in $(seq 600); do [ -e "$state/loading" ] || exit 0; sleep 0.05; done
    exit 1 ;;
esac
exit 2
"""


def unit_lines(text: str, key: str) -> list:
    """Each KEY= line of the unit's [Service] section, its value as systemd reads it: the lines a backslash
    ends joined, $$ a $."""
    service = text.split("[Service]", 1)[1].split("\n[", 1)[0].replace("\\\n", " ")
    return [line[len(key) + 1:].replace("$$", "$") for line in service.splitlines() if line.startswith(key + "=")]


class Unit:
    def __init__(self, name: str):
        self.name = name
        self.text = (REPO / f"{name}.service.template").read_text()
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="vllm-stop-"))
        self.state = self.t / "state"
        self.state.mkdir()
        self.docker = self.t / "docker"
        self.docker.write_text(FAKE_DOCKER)
        self.docker.chmod(0o755)
        home = self.t / "home"
        (home / ".config" / "qwen38").mkdir(parents=True)
        # the check of the lane's files at the head of flash-local's ExecStart: there, and passed
        (home / ".config" / "qwen38" / "flash-local-files.sh").write_text("exit 0\n")
        self.text = self.text.replace("__HOME__", str(home))
        # systemd makes the unit's runtime directory, empty, at each start, and names it to every command
        self.run_dir = self.t / "run" / name
        self.run_dir.mkdir(parents=True)
        self.env = {"PATH": "/usr/bin:/bin", "RUNTIME_DIRECTORY": str(self.run_dir),
                    "FAKE_DOCKER_STATE": str(self.state)}

    def argv(self, line: str) -> list:
        argv = shlex.split(line[1:] if line.startswith("-") else line)
        return [a.replace("/usr/bin/docker", str(self.docker)) for a in argv]

    def start(self) -> subprocess.Popen:
        (start,) = unit_lines(self.text, "ExecStart")
        p = subprocess.Popen(self.argv(start), env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(600):
            if (self.state / "loading").exists() or p.poll() is not None:
                break
            time.sleep(0.05)
        assert (self.state / "loading").exists(), f"{self.name}: the engine never started: {p.stderr.read()}"
        return p

    def stop(self):
        for line in unit_lines(self.text, "ExecStop"):
            r = subprocess.run(self.argv(line), env=self.env, capture_output=True, timeout=60)
            if r.returncode and not line.startswith("-"):
                break                   # systemd runs no ExecStop line after one that failed unprefixed

    def success(self) -> set:
        return {int(c) for line in unit_lines(self.text, "SuccessExitStatus") for c in line.split()} | {0}


class AStopWhileTheEngineLoads(unittest.TestCase):
    def setUp(self):
        self.units = [Unit(name) for name in VLLM_UNITS]

    def tearDown(self):
        for u in self.units:
            shutil.rmtree(u.t, ignore_errors=True)

    def test_a_stop_asked_while_the_engine_loads_ends_clean(self):
        for u in self.units:
            with self.subTest(unit=u.name):
                p = u.start()
                u.stop()
                code = p.wait(timeout=60)
                self.assertEqual(code, 143, f"{u.name}: a plain stop left the main process's {code}")
                self.assertIn(code, u.success())

    def test_an_engine_that_dies_on_its_own_still_fails(self):
        # nothing asked the stop: the engine's 1 is the unit's, which systemd counts a failure and restarts
        # (Restart=always); flash-local's 78 for files missing keeps its meaning (RestartPreventExitStatus)
        for u in self.units:
            with self.subTest(unit=u.name):
                p = u.start()
                (u.state / "end").touch()
                code = p.wait(timeout=60)
                self.assertEqual(code, 1)
                self.assertNotIn(code, u.success())

    def test_the_stop_marks_itself_before_it_stops_the_engine(self):
        # the mark goes in the unit's own runtime directory, which systemd empties when the unit stops and
        # again before an automatic restart (RuntimeDirectoryPreserve=no, its default), so that a crash
        # (after which systemd runs ExecStop too) leaves no mark to the next start
        for u in self.units:
            with self.subTest(unit=u.name):
                self.assertEqual(unit_lines(u.text, "RuntimeDirectory"), [u.name])
                self.assertEqual(unit_lines(u.text, "RuntimeDirectoryPreserve"), [])
                stops = unit_lines(u.text, "ExecStop")
                self.assertEqual(len(stops), 2, stops)
                self.assertTrue(stops[0].startswith("-") and "$RUNTIME_DIRECTORY/stopping" in stops[0], stops)
                self.assertEqual(stops[1], f"-/usr/bin/docker stop -t 20 {u.name}")


if __name__ == "__main__":
    unittest.main()
