/**
 * Procedural Asset Provider
 *
 * Generates NPC character meshes from Three.js primitives.
 * This is the default (and LOD-0 fallback) implementation of
 * the AssetProvider interface.
 *
 * When a GLTFAssetProvider is added in the future, this provider
 * continues to serve as the fallback for distant/low-tier NPCs
 * and as the guaranteed-available option when no models are loaded.
 */

import * as THREE from 'three';
import { AssetProvider } from './asset_contract.js';


// ---------- Colour palette ----------

const SKIN_COLOURS = [0xf5d0a9, 0xe0c097, 0xc49e6b, 0xa47149, 0x8d5524];

const OCCUPATION_COLOURS = {
    blacksmith:    0x444444,
    farmer:        0x6b8e23,
    merchant:      0x8b6914,
    tavern_keeper: 0x8b4513,
    priest:        0xf5f5dc,
    guard:         0x4a5568,
    labourer:      0x7b6b4f,
    traveller:     0x2266aa,
};

const DEFAULT_CLOTHING_COLOUR = 0x6b6b6b;


// ---------- Shared geometries ----------

const _bodyGeo = new THREE.CylinderGeometry(0.15, 0.18, 0.5, 8);
const _headGeo = new THREE.SphereGeometry(0.13, 8, 6);
const _hatGeo = new THREE.ConeGeometry(0.15, 0.15, 8);


// ---------- Provider ----------

export class ProceduralAssetProvider extends AssetProvider {
    /** @override */
    getCharacterAsset(archetype, options = {}) {
        const npcId = options.npc_id || '';
        const occupation = options.occupation || archetype;

        const clothingColour = OCCUPATION_COLOURS[occupation] || DEFAULT_CLOTHING_COLOUR;
        const skinColour = SKIN_COLOURS[_hashCode(npcId) % SKIN_COLOURS.length];

        const group = new THREE.Group();

        // Body (tapered cylinder)
        const bodyMat = new THREE.MeshStandardMaterial({
            color: clothingColour,
            roughness: 0.8,
        });
        const body = new THREE.Mesh(_bodyGeo, bodyMat);
        body.position.y = 0;
        body.castShadow = true;
        group.add(body);

        // Head (sphere)
        const headMat = new THREE.MeshStandardMaterial({
            color: skinColour,
            roughness: 0.6,
        });
        const head = new THREE.Mesh(_headGeo, headMat);
        head.position.y = 0.38;
        head.castShadow = true;
        group.add(head);

        // Occupation indicator (hat/accessory)
        const hat = _createOccupationIndicator(occupation, clothingColour);
        if (hat) {
            hat.position.y = 0.52;
            group.add(hat);
        }

        // Standard attachment points
        const attachments = {
            head_top:   { x: 0, y: 0.52, z: 0 },
            right_hand: { x: 0.2, y: 0.05, z: 0 },
            left_hand:  { x: -0.2, y: 0.05, z: 0 },
            back:       { x: 0, y: 0.1, z: -0.15 },
            label:      { x: 0, y: 0.75, z: 0 },
            speech:     { x: 0, y: 0.9, z: 0 },
        };

        const animation = {
            idle_bob_speed: 2.0,
            idle_bob_amplitude: 0.02,
            walk_bob_speed: 6.0,
            walk_bob_amplitude: 0.04,
            base_y: 0.25,
        };

        return {
            archetype_id: archetype,
            source_type: 'procedural',
            mesh: group,
            attachments,
            animation,
            bounds: { width: 0.36, height: 0.9, depth: 0.36 },
            lod_level: 0,
        };
    }

    /** @override */
    dispose() {
        // Shared geometries are module-level singletons — no cleanup needed
    }
}


// ---------- Helpers ----------

function _createOccupationIndicator(occupation) {
    switch (occupation) {
        case 'blacksmith': {
            const mat = new THREE.MeshStandardMaterial({ color: 0x333333 });
            return new THREE.Mesh(_hatGeo, mat);
        }
        case 'priest': {
            const geo = new THREE.CylinderGeometry(0.08, 0.1, 0.15, 6);
            const mat = new THREE.MeshStandardMaterial({ color: 0xffffff });
            return new THREE.Mesh(geo, mat);
        }
        case 'guard': {
            const geo = new THREE.SphereGeometry(0.14, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2);
            const mat = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.6 });
            return new THREE.Mesh(geo, mat);
        }
        case 'farmer': {
            const geo = new THREE.ConeGeometry(0.2, 0.08, 8);
            const mat = new THREE.MeshStandardMaterial({ color: 0xd4a946 });
            return new THREE.Mesh(geo, mat);
        }
        case 'traveller': {
            // Wide-brim travel hat
            const geo = new THREE.CylinderGeometry(0.2, 0.22, 0.06, 8);
            const mat = new THREE.MeshStandardMaterial({ color: 0x3a2211 });
            const brim = new THREE.Mesh(geo, mat);
            const crownGeo = new THREE.CylinderGeometry(0.08, 0.1, 0.12, 8);
            const crown = new THREE.Mesh(crownGeo, mat);
            crown.position.y = 0.08;
            brim.add(crown);
            return brim;
        }
        default:
            return null;
    }
}

function _hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash);
}
