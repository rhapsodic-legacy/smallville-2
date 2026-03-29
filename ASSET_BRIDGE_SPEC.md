# Asset Bridge Specification

> Cross-project alignment document for the 3D Asset Forge, Smallville 2,
> and the AI Game Studio (claude_agent_swarm). Give this to any Claude
> instance working on any of these projects so it understands the
> shared contract.

---

## The Problem

All three projects (Smallville 2, AI Game Studio's rpg_game, and the
planned 3D Asset Forge) currently render characters as **procedural
Three.js primitives** — cylinders for bodies, spheres for heads,
cones for hats. There are no external 3D assets, no glTF files, no
texture maps.

The long-term goal is PS1-era character models (300–1,500 polys,
hand-painted 64×256 textures, ~20-bone skeletons), eventually
upgrading to PS2 fidelity. But the procedural primitives must
continue working as the default fallback and LOD-0 representation.

## The Solution: AssetProvider Interface

Rather than ripping out procedural rendering and replacing it with
glTF loading, we've introduced an **abstraction layer** that lets
both approaches coexist behind the same interface.

### Architecture

```
Server (Python)
  NPC.to_dict() includes "archetype" field
      │
      ▼
WebSocket JSON: { "archetype": "blacksmith", "occupation": "blacksmith", ... }
      │
      ▼
Client (Three.js)
  NPCRenderer holds an AssetProvider
      │
      ├── ProceduralAssetProvider (current, built-in, LOD-0)
      │     Returns composed Three.js primitives
      │
      └── GLTFAssetProvider (future, from Asset Forge)
            Loads glTF models, returns same interface
```

The renderer calls `assetProvider.getCharacterAsset(archetype, options)`
and receives a `CharacterAsset` object. It never knows whether the
geometry was procedurally generated or loaded from a file.

### CharacterAsset Contract

Every asset provider must return objects matching this shape:

```javascript
{
  archetype_id: string,              // "blacksmith", "farmer_male", etc.
  source_type: "procedural" | "gltf",
  mesh: THREE.Group,                 // The character mesh (body + parts)

  attachments: {
    head_top:   { x, y, z },         // Hat/helmet mount point
    right_hand: { x, y, z },         // Weapon/tool mount
    left_hand:  { x, y, z },         // Shield/item mount
    back:       { x, y, z },         // Pack/cloak mount
    label:      { x, y, z },         // Name label position
    speech:     { x, y, z },         // Speech bubble position
  },

  animation: {
    idle_bob_speed: number,           // Oscillation speed when idle
    idle_bob_amplitude: number,       // Vertical displacement when idle
    walk_bob_speed: number,           // Oscillation speed when walking
    walk_bob_amplitude: number,       // Vertical displacement when walking
    base_y: number,                   // Resting Y position above ground
  },

  bounds: { width, height, depth },   // Bounding box in world units
  lod_level: number,                  // 0=procedural, 1=low-poly, 2=mid
}
```

### AssetProvider Interface

```javascript
class AssetProvider {
  getCharacterAsset(archetype, options) → CharacterAsset
  dispose()  // clean up cached resources
}
```

**Options passed to the provider:**
```javascript
{
  npc_id: string,           // For deterministic colour/variation hashing
  occupation: string,       // "blacksmith", "farmer", etc.
  cognition_tier: number,   // 1-4 (can be used for LOD decisions)
}
```

### Current Implementations

| Provider | Status | Output |
|----------|--------|--------|
| `ProceduralAssetProvider` | **Built, live** | Three.js primitives (cylinders, spheres, cones) |
| `GLTFAssetProvider` | **Not yet built** | Would load `.glb` files from asset forge |

### How the Renderer Uses Assets

The NPCRenderer (in both Smallville 2 and rpg_game) does this:

1. Receives NPC data from server with `archetype` field
2. Calls `assetProvider.getCharacterAsset(archetype, options)`
3. Gets back a `CharacterAsset` with mesh, attachment points, animation config
4. Attaches name label at `attachments.label`
5. Attaches speech bubble at `attachments.speech`
6. Uses `animation` config for idle bob, walk bob, base Y position
7. Handles interpolation, direction, activity indicators itself

The provider handles **only** the character mesh and its metadata.
The renderer handles everything else (labels, bubbles, indicators,
interpolation, direction).

---

## Server-Side: Archetype Field

### Smallville 2

The `NPC` data model now includes an `archetype` field:

```python
@dataclass
class NPC:
    occupation: str        # "blacksmith", "farmer", etc.
    archetype: str = ""    # Visual archetype — defaults to occupation if empty
```

`NPC.to_dict()` sends:
```json
{ "archetype": "blacksmith", "occupation": "blacksmith", ... }
```

The archetype is currently derived from occupation, but the field
exists so the asset forge can target richer archetypes later
(e.g. `"blacksmith_male_elderly"`, `"farmer_female_young"`).

### AI Game Studio (rpg_game)

The rpg_game NPC data in `npc_data.py` would need a similar
`archetype` field added to NPC definitions. The client renderer
would need the same AssetProvider refactor — swap inline Three.js
primitive creation for a provider call.

**What needs to change in rpg_game:**
1. Add `archetype` field to NPC data dicts in `npc_data.py`
2. Include it in the WebSocket state broadcast
3. Refactor the client's NPC rendering to use the AssetProvider pattern
4. Import `ProceduralAssetProvider` as the default (preserves current look)

The contract is identical across both projects — an asset produced
for Smallville 2 works in rpg_game and vice versa.

---

## For the 3D Asset Forge

### What to produce

The asset forge pipeline should output **CharacterAsset-compatible
bundles** for each archetype. Concretely:

```
output/
  blacksmith/
    blacksmith.glb          — glTF binary (mesh + skeleton + textures)
    blacksmith.meta.json    — attachment points, animation config, bounds
  farmer_male/
    farmer_male.glb
    farmer_male.meta.json
  ...
```

### meta.json schema

```json
{
  "archetype_id": "blacksmith",
  "lod_level": 1,
  "bounds": { "width": 0.36, "height": 0.9, "depth": 0.36 },
  "attachments": {
    "head_top":   { "x": 0, "y": 0.52, "z": 0 },
    "right_hand": { "x": 0.2, "y": 0.05, "z": 0 },
    "left_hand":  { "x": -0.2, "y": 0.05, "z": 0 },
    "back":       { "x": 0, "y": 0.1, "z": -0.15 },
    "label":      { "x": 0, "y": 0.75, "z": 0 },
    "speech":     { "x": 0, "y": 0.9, "z": 0 }
  },
  "animation": {
    "idle_bob_speed": 2.0,
    "idle_bob_amplitude": 0.02,
    "walk_bob_speed": 6.0,
    "walk_bob_amplitude": 0.04,
    "base_y": 0.25
  }
}
```

### Fidelity targets

| Era | Poly budget | Texture | Bones | Notes |
|-----|-------------|---------|-------|-------|
| PS1 (Phase 1) | 300–1,500 | 64×64 to 256×256 diffuse | ~20 | FF7-9 field models |
| PS2 (Phase 2) | 5k–15k | 512×512, simple normals | ~40 + facial | Same pipeline, higher params |
| PS3+ (Phase 3) | 30k+ | 1024+, normal/spec maps | 60+ blend shapes | Different project entirely |

### GLTFAssetProvider (to be built)

When the forge produces `.glb` files, a `GLTFAssetProvider` wraps them:

```javascript
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { AssetProvider } from './asset_contract.js';

class GLTFAssetProvider extends AssetProvider {
  constructor(assetBasePath = '/static/assets/characters/') {
    super();
    this.loader = new GLTFLoader();
    this.cache = new Map();  // archetype -> CharacterAsset
    this.basePath = assetBasePath;
  }

  async preload(archetypes) {
    // Load all needed archetypes at init
  }

  getCharacterAsset(archetype, options) {
    if (this.cache.has(archetype)) {
      return this._cloneAsset(this.cache.get(archetype));
    }
    // Fallback to procedural if model not loaded
    return this.fallback.getCharacterAsset(archetype, options);
  }
}
```

Key design: the GLTF provider holds a `ProceduralAssetProvider` as
fallback. If a model isn't loaded for an archetype, it falls back
to procedural. This means:
- The game always works, even with zero models loaded
- Models can be added incrementally (blacksmith first, then farmer, etc.)
- Distant NPCs (tier 3-4) can use procedural LOD-0 even when models exist

### LOD strategy

```
Tier 1 (near camera):  GLTFAssetProvider → full model (LOD 1-2)
Tier 2 (medium):       GLTFAssetProvider → simplified model (LOD 1)
Tier 3 (far):          ProceduralAssetProvider → primitives (LOD 0)
Tier 4 (frozen):       Not rendered
```

The renderer already receives `cognition_tier` in the NPC data.
The provider can use it to select LOD level.

---

## Coordinate System & Scale (shared across all projects)

| Property | Value |
|----------|-------|
| Ground plane | (x, z) |
| Up axis | y |
| Tile size | 1 world unit |
| Character height | ~0.9 units (head top) |
| Character width | ~0.36 units |
| Base Y position | 0.25 above ground |

All models from the asset forge must match this scale.
A character standing on a tile should have their feet at y=0
and the base_y offset handles the visual lift.

---

## File Locations

### Smallville 2
```
client/js/asset_contract.js      — AssetProvider interface + CharacterAsset typedef
client/js/procedural_assets.js   — ProceduralAssetProvider (default, LOD-0)
client/js/npc_renderer.js        — NPCRenderer (consumes AssetProvider)
core/npc/models.py               — NPC.archetype field + to_dict()
```

### AI Game Studio (rpg_game) — needs updating
```
rpg-game/client/js/              — Needs same AssetProvider refactor
rpg-game/server/npc_data.py      — Needs archetype field in NPC dicts
```

### 3D Asset Forge (new repo)
```
pipeline/07-export/              — Must output .glb + .meta.json per archetype
quality/validators/              — Must validate against CharacterAsset schema
```

---

## Summary for Each Claude Instance

### If you're working on Smallville 2:
The AssetProvider interface is live. `ProceduralAssetProvider` is the
default. When the forge produces models, you'll add a `GLTFAssetProvider`
that loads them and pass it to `NPCRenderer(scene, provider)`. The
`archetype` field flows from server to client in the NPC data.

### If you're working on the AI Game Studio:
The same refactor needs to happen in rpg_game's client. Add `archetype`
to NPC data, refactor the Three.js NPC rendering behind the same
`AssetProvider` interface, use `ProceduralAssetProvider` as default.
The contract schema is identical — assets are cross-compatible.

### If you're working on the 3D Asset Forge:
Your pipeline must output `.glb` + `.meta.json` bundles that conform
to the CharacterAsset contract above. Attachment points, animation
config, and bounds must match the schema exactly. Scale must match
the coordinate system table. The consuming renderers will load your
output via `GLTFAssetProvider` which falls back to procedural for
any missing archetype.
