# client/ — Three.js Frontend

## Purpose
Thin rendering client. Receives world state from server via WebSocket,
renders 3D scene with Three.js. Sends player inputs back to server.
No game logic — display only.

## Architecture
- **index.html** — Entry point, loads JS/CSS
- **js/** — Modules split by concern
  - asset_contract.js — AssetProvider interface + CharacterAsset typedef
  - procedural_assets.js — ProceduralAssetProvider (default, LOD-0 fallback)
  - npc_renderer.js — NPC meshes via pluggable AssetProvider, movement, indicators
  - world_renderer.js — Terrain, buildings, objects
  - player_controls.js — WASD movement, camera, interaction
  - chat_ui.js — Text input, conversation display
  - trade_ui.js — Trading interface
  - hud.js — Time, gold, minimap, notifications
  - websocket.js — Connection management, message handling
- **css/** — Styles for UI overlays

## 3D Conventions
- Three.js via CDN (no build step)
- Coordinate system: (x, z) ground plane, y is up
- Tile size: 1 unit, character height ~0.9 units
- Camera: perspective, angled top-down, follows player

## Asset Provider Pattern
NPC visuals are produced by a pluggable **AssetProvider**. The renderer
calls `getCharacterAsset(archetype, options)` and receives a CharacterAsset
with mesh, attachment points, animation config, and bounds. It never knows
whether the geometry is procedural or loaded from glTF.

- **ProceduralAssetProvider** (current default): Three.js primitives, always available
- **GLTFAssetProvider** (future): loads .glb from 3D Asset Forge, falls back to procedural
- See ASSET_BRIDGE_SPEC.md for the full contract and cross-project alignment

## Rendering Pipeline
1. Server sends state JSON via WebSocket
2. Client parses state
3. Clear and rebuild scene (or diff-update for performance)
4. Render terrain tiles as coloured geometry
5. Render buildings as procedural meshes
6. Render NPCs as stylised character meshes
7. Update HUD overlays
8. Animate NPC movement and activities

## Aligns With
- rpg_game in claude_agent_swarm uses same Three.js + WebSocket pattern
- Same coordinate system and tile conventions
- Same server-authoritative thin-client philosophy
