#!/usr/bin/env python3
"""
Dump every NPC's memory from the running Smallville server.

Hits `/api/memory/dump` on localhost:8002 (override with
`--host` / `--port`) and writes a human-readable log plus the raw
JSON to disk. Meant so Claude Code can see exactly what every NPC
"remembers" — the verbatim transcripts, reflections, outcomes,
commitments, accusations, agenda events — when a bug report
surfaces.

Run:
    python3 tools/dump_memories.py                    # prints to stdout
    python3 tools/dump_memories.py -o memories.txt    # writes to file
    python3 tools/dump_memories.py --npc dara         # one NPC only
    python3 tools/dump_memories.py --include-compacted

The server must be running:  PYTHONPATH=. python3 server/main.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _fetch(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"ERROR: could not reach {url}: {e}", file=sys.stderr)
        print(
            "Is the server running? "
            "Start with: PYTHONPATH=. python3 server/main.py",
            file=sys.stderr,
        )
        sys.exit(2)


def _format_memory(mem: dict[str, Any]) -> str:
    cat = (mem.get("category") or "?").upper()
    imp = mem.get("importance", 0.0)
    dots = "●" * int(round(imp * 5)) + "○" * (5 - int(round(imp * 5)))
    gt = mem.get("game_time", 0.0)
    day = int(gt // 1440)
    minute = int(gt % 1440)
    hh, mm = divmod(minute, 60)
    tag_list = mem.get("tags") or []
    tags = f"  [tags: {', '.join(tag_list)}]" if tag_list else ""
    desc = mem.get("description", "")
    return (
        f"{cat:<18} {dots}  Day {day} {hh:02d}:{mm:02d}  "
        f"{desc}{tags}"
    )


def _format_npc(npc_id: str, data: dict[str, Any]) -> str:
    lines: list[str] = []
    name = data.get("name", npc_id)
    occ = data.get("occupation", "")
    pos = data.get("position", {})
    home = data.get("home", {})
    lines.append("=" * 80)
    lines.append(
        f"{name} ({occ}, id={npc_id}) — "
        f"pos=({pos.get('x')}, {pos.get('z')}) "
        f"home=({home.get('x')}, {home.get('z')}) "
        f"tier={data.get('cognition_tier')} "
        f"activity={data.get('activity')}"
    )
    action = data.get("current_action") or ""
    if action:
        lines.append(f"  Current: {action}")
    goals = data.get("goals") or []
    if goals:
        lines.append("  Goals:")
        for g in goals:
            status = g.get("status", "?")
            lines.append(f"    - [{status}] {g.get('description', '')}")
    lines.append("-" * 80)
    mems = data.get("recent_memories") or []
    lines.append(f"  Memories ({len(mems)}):")
    for mem in mems:
        lines.append("    " + _format_memory(mem))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8002,
        help="Server port (default: 8002)",
    )
    parser.add_argument(
        "--npc", default=None,
        help="Dump only a single NPC (matches name substring or full id)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Cap memories per NPC (0 = everything, default)",
    )
    parser.add_argument(
        "--include-compacted", action="store_true",
        help="Include tombstoned raw memories alongside summaries",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write output to file (default: stdout)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON rather than formatted log",
    )
    args = parser.parse_args()

    qs = []
    if args.limit:
        qs.append(f"limit={args.limit}")
    if args.include_compacted:
        qs.append("include_compacted=true")
    query = ("?" + "&".join(qs)) if qs else ""
    url = f"http://{args.host}:{args.port}/api/memory/dump{query}"
    data = _fetch(url)

    if args.npc:
        needle = args.npc.lower()
        filtered = {
            nid: d for nid, d in data.get("npcs", {}).items()
            if needle in nid.lower()
            or needle in (d.get("name") or "").lower()
        }
        data["npcs"] = filtered
        if not filtered:
            print(
                f"No NPC matching {args.npc!r}. "
                f"Available: {list(data.get('npcs', {}).keys())}",
                file=sys.stderr,
            )
            return 1

    if args.json:
        text = json.dumps(data, indent=2)
    else:
        header = (
            f"# Memory dump — "
            f"Day {data.get('day')} {data.get('time')} "
            f"({data.get('phase')})\n"
        )
        bodies = [
            _format_npc(nid, d)
            for nid, d in data.get("npcs", {}).items()
        ]
        text = header + "\n\n".join(bodies) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.output} ({len(text):,} bytes).")
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
