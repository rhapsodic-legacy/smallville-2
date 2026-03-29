/**
 * Character Asset Contract
 *
 * Defines the interface between asset providers and the NPC renderer.
 * Any asset source — procedural geometry, loaded glTF, or a future
 * 3D asset forge — must produce objects conforming to this contract.
 *
 * The renderer consumes CharacterAsset objects and never knows
 * (or cares) how the geometry was produced.
 */

/**
 * @typedef {Object} AttachmentPoints
 * @property {{x: number, y: number, z: number}} head_top    — hat/helmet mount
 * @property {{x: number, y: number, z: number}} right_hand  — weapon/tool mount
 * @property {{x: number, y: number, z: number}} left_hand   — shield/item mount
 * @property {{x: number, y: number, z: number}} back        — pack/cloak mount
 * @property {{x: number, y: number, z: number}} label       — name label position
 * @property {{x: number, y: number, z: number}} speech       — speech bubble position
 */

/**
 * @typedef {Object} AnimationConfig
 * @property {number} idle_bob_speed     — oscillation speed when idle
 * @property {number} idle_bob_amplitude — vertical displacement when idle
 * @property {number} walk_bob_speed     — oscillation speed when walking
 * @property {number} walk_bob_amplitude — vertical displacement when walking
 * @property {number} base_y             — resting Y position
 */

/**
 * @typedef {Object} CharacterAsset
 * @property {string} archetype_id       — e.g. "blacksmith_male", "farmer_female"
 * @property {"procedural"|"gltf"} source_type
 * @property {THREE.Group} mesh          — the character mesh group (body + parts)
 * @property {AttachmentPoints} attachments
 * @property {AnimationConfig} animation
 * @property {{width: number, height: number, depth: number}} bounds
 * @property {number} lod_level          — 0=procedural fallback, 1=low-poly, 2=mid
 */

/**
 * AssetProvider interface.
 *
 * Implementations produce CharacterAsset objects from archetype specs.
 * The NPC renderer holds one provider and calls getCharacterAsset()
 * for each NPC. Swapping the provider swaps the entire visual style.
 *
 * Built-in: ProceduralAssetProvider (current primitives)
 * Future:   GLTFAssetProvider (loaded 3D models from asset forge)
 */
export class AssetProvider {
    /**
     * Produce a character mesh for the given archetype.
     *
     * @param {string} archetype — archetype identifier (e.g. "blacksmith")
     * @param {Object} options
     * @param {string} options.npc_id   — for deterministic colour hashing
     * @param {string} options.occupation
     * @param {number} options.cognition_tier — for LOD decisions
     * @returns {CharacterAsset}
     */
    getCharacterAsset(archetype, options) {
        throw new Error('AssetProvider.getCharacterAsset() not implemented');
    }

    /**
     * Clean up any cached resources (textures, geometries).
     * Called when the renderer is destroyed or provider is swapped.
     */
    dispose() {}
}
