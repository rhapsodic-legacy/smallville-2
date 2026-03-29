"""
WebSocket contract tests.

Connects to the live server and verifies that messages contain
the required fields. Catches stale servers, missing subsystems,
and schema regressions.

Run with: pytest tests/integration/test_websocket_contract.py -m integration
Requires: server running on localhost:8002
"""

import json
import pytest

try:
    from websockets.sync.client import connect as ws_connect
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


SERVER_URL = "ws://localhost:8002/ws"
HEALTH_URL = "http://localhost:8002/health"


def _server_reachable() -> bool:
    """Check if the server is running."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8002/health", timeout=2)
        return True
    except Exception:
        return False


skip_if_no_server = pytest.mark.skipif(
    not _server_reachable(),
    reason="Server not running on localhost:8002",
)
pytestmark = pytest.mark.integration


@skip_if_no_server
class TestHealthEndpoint:
    def test_health_returns_ok(self):
        import urllib.request
        resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
        data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_health_reports_npcs(self):
        import urllib.request
        resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
        data = json.loads(resp.read())
        assert data["npcs"] > 0, "Server has no NPCs loaded"

    def test_health_reports_buildings(self):
        import urllib.request
        resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
        data = json.loads(resp.read())
        assert data["buildings"] > 0, "Server has no buildings loaded"


@skip_if_no_server
@pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets package not installed")
class TestInitMessage:
    def test_init_has_world(self):
        with ws_connect(SERVER_URL) as ws:
            msg = json.loads(ws.recv(timeout=5))
            assert msg["type"] == "init"
            assert "world" in msg, "Init message missing 'world'"

    def test_init_has_time(self):
        with ws_connect(SERVER_URL) as ws:
            msg = json.loads(ws.recv(timeout=5))
            assert "time" in msg, "Init message missing 'time'"
            assert "day" in msg["time"]
            assert "phase" in msg["time"]

    def test_init_has_npcs(self):
        with ws_connect(SERVER_URL) as ws:
            msg = json.loads(ws.recv(timeout=5))
            assert "npcs" in msg, "Init message missing 'npcs' — stale server?"
            assert len(msg["npcs"]) > 0, "Init message has empty NPC list"

    def test_init_npcs_have_required_fields(self):
        required = {"npc_id", "name", "x", "z", "activity", "occupation"}
        with ws_connect(SERVER_URL) as ws:
            msg = json.loads(ws.recv(timeout=5))
            for npc in msg["npcs"]:
                missing = required - set(npc.keys())
                assert not missing, f"NPC {npc.get('npc_id', '?')} missing: {missing}"

    def test_init_has_buildings(self):
        with ws_connect(SERVER_URL) as ws:
            msg = json.loads(ws.recv(timeout=5))
            assert "buildings" in msg, "Init message missing 'buildings'"
            assert len(msg["buildings"]) > 0


@skip_if_no_server
@pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets package not installed")
class TestTickMessage:
    def test_tick_has_time(self):
        with ws_connect(SERVER_URL) as ws:
            ws.recv(timeout=5)  # skip init
            tick = json.loads(ws.recv(timeout=5))
            assert tick["type"] == "tick"
            assert "time" in tick

    def test_tick_has_npcs(self):
        with ws_connect(SERVER_URL) as ws:
            ws.recv(timeout=5)  # skip init
            tick = json.loads(ws.recv(timeout=5))
            assert "npcs" in tick, "Tick message missing 'npcs'"
            assert len(tick["npcs"]) > 0

    def test_ping_pong(self):
        with ws_connect(SERVER_URL) as ws:
            ws.recv(timeout=5)  # skip init
            ws.send(json.dumps({"type": "ping"}))
            resp = json.loads(ws.recv(timeout=5))
            # Might get a tick first, keep reading
            while resp.get("type") == "tick":
                resp = json.loads(ws.recv(timeout=5))
            assert resp["type"] == "pong"
