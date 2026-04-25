"""
NPC schedule-dispatch latency regression test.

When the simulation starts, NPCs have a template schedule anchored
at action_start_minutes=0. The first cognition_tick is supposed to
anchor each NPC's timer, dispatch them to their first schedule
entry, and let the movement loop take them from there.

Regression: for a period we routed `task_decompose` through the
LLM by default. That caused the first cognition_tick to make 8
sequential LLM calls (one per NPC) before ANY NPC started moving.
On local Gemma that was ~4 real minutes of the town sitting
completely frozen in their homes — exactly the "NPCs in huts at
9AM" bug the user reported.

This test boots a real server with MockProvider (no network calls)
and measures how long it takes for every NPC to leave
schedule_index=0. With deterministic task_decompose it should be
well under a few seconds.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

TEST_PORT = int(os.environ.get("SMALLVILLE_DISPATCH_PORT", "8913"))


def _write_patched_main() -> Path:
    patched = Path("/tmp") / f"smallville_dispatch_{TEST_PORT}.py"
    patched.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "os.environ['ANTHROPIC_API_KEY'] = ''\n"
        "os.environ['MISTRAL_API_KEY'] = ''\n"
        "import core.npc.gemma_provider as _gp\n"
        "_gp.ollama_available = lambda *a, **k: False\n"
        "from server import main as sm\n"
        "import uvicorn\n"
        f"uvicorn.run(sm.app, host='127.0.0.1', port={TEST_PORT}, log_level='warning')\n"
    )
    return patched


def _start_server() -> tuple[subprocess.Popen, Path]:
    log_path = Path("/tmp") / f"smallville_dispatch_{TEST_PORT}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    env["ANTHROPIC_API_KEY"] = ""
    env["MISTRAL_API_KEY"] = ""
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(_write_patched_main())],
        cwd=str(REPO),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_path


async def _wait_ready() -> bool:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{TEST_PORT}/health", timeout=1,
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return False


def _fetch_npcs() -> list[dict]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{TEST_PORT}/api/debug/npcs", timeout=5,
    ) as resp:
        return json.loads(resp.read())["npcs"]


async def test_all_npcs_anchor_quickly() -> None:
    """
    Every NPC must anchor their action_start_minutes (i.e. receive
    their first dispatch) within 3 real seconds of server start.

    Regression: when task_decompose defaulted to LLM routing, the
    first cognition_tick blocked on 8 sequential Gemma calls and
    NPCs sat frozen for minutes before any of them was dispatched.
    The check is a wall-clock budget: a fresh server should have
    every NPC anchored almost immediately with deterministic
    decomposition.
    """
    proc, log_path = _start_server()
    try:
        assert await _wait_ready(), (
            f"Server did not become ready. Log:\n{log_path.read_text()[-1500:]}"
        )

        deadline = time.time() + 5
        while time.time() < deadline:
            npcs = _fetch_npcs()
            real_npcs = [n for n in npcs if n.get("npc_id") != "player"]
            not_anchored = [
                n for n in real_npcs
                if (n.get("action_start_minutes") or 0.0) == 0.0
            ]
            if not not_anchored:
                print(f"All {len(real_npcs)} NPCs anchored.")
                break
            await asyncio.sleep(0.3)
        else:
            still_zero = [
                (n["name"], n.get("action_start_minutes"))
                for n in _fetch_npcs()
                if n.get("npc_id") != "player"
                and (n.get("action_start_minutes") or 0.0) == 0.0
            ]
            raise AssertionError(
                f"After 5s, {len(still_zero)} NPC(s) still not anchored: "
                f"{still_zero}. Dispatch is starved — LLM on the hot path."
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def test_all_npcs_advance_past_breakfast() -> None:
    """
    Within ~70 real seconds (breakfast=60 game-min, clock=1.2x speed),
    every NPC must have advanced past schedule_index=0. This catches
    the scenario the user reported directly: NPCs visibly sitting in
    their huts long after dawn instead of dispatching to work.
    """
    proc, log_path = _start_server()
    try:
        assert await _wait_ready()

        deadline = time.time() + 75
        start = time.time()
        while time.time() < deadline:
            npcs = _fetch_npcs()
            real_npcs = [n for n in npcs if n.get("npc_id") != "player"]
            still_at_zero = [
                n for n in real_npcs
                if (n.get("schedule_index") or 0) == 0
            ]
            if not still_at_zero:
                elapsed = time.time() - start
                print(
                    f"All {len(real_npcs)} NPCs advanced past idx=0 "
                    f"in {elapsed:.1f}s"
                )
                return
            await asyncio.sleep(1.0)

        still_at_zero = [
            (n["name"], n.get("current_action", "")[:40])
            for n in _fetch_npcs()
            if n.get("npc_id") != "player"
            and (n.get("schedule_index") or 0) == 0
        ]
        raise AssertionError(
            f"After {time.time() - start:.0f}s, {len(still_at_zero)} NPC(s) "
            f"still on schedule_index=0 (breakfast): {still_at_zero}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _run_all():
    tests = [
        ("all npcs anchor quickly", test_all_npcs_anchor_quickly),
        ("all npcs advance past breakfast", test_all_npcs_advance_past_breakfast),
    ]
    fails = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            await fn()
            print(f"  PASS")
        except AssertionError as e:
            fails.append((name, str(e)))
            print(f"  FAIL: {e}")
        except Exception as e:
            fails.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR: {e}")

    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
