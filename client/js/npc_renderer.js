/**
 * Smallville 2 — NPC Renderer
 *
 * Stanford approach: server sends tile position + trail (tiles traversed
 * this tick, max 2-3). Client walks through the trail waypoints smoothly.
 * No full path, no complex sync state, no snap-back.
 */

import * as THREE from 'three';
import { ProceduralAssetProvider } from './procedural_assets.js';

// If the NPC needs to cover more than this distance, teleport instantly.
const TELEPORT_DISTANCE = 4.0;

// Tick interval (must match server TICK_INTERVAL)
const TICK_INTERVAL = 1.0;

// Arrive at trail end slightly before next tick
const ARRIVAL_FACTOR = 0.85;


// ---------- NPC Renderer ----------

export class NPCRenderer {
    /**
     * @param {THREE.Scene} scene
     * @param {import('./asset_contract.js').AssetProvider} [assetProvider]
     */
    constructor(scene, assetProvider = null) {
        this.scene = scene;
        this.assetProvider = assetProvider || new ProceduralAssetProvider();
        this.npcGroup = new THREE.Group();
        this.npcGroup.name = 'npc-group';
        this.scene.add(this.npcGroup);

        /** @type {Map<string, object>} */
        this.npcMeshes = new Map();

        /**
         * Per-NPC movement state: trail waypoints and walk speed.
         * @type {Map<string, object>}
         */
        this.moveStates = new Map();

        // Name label canvas — reused
        this._labelCanvas = document.createElement('canvas');
        this._labelCanvas.width = 256;
        this._labelCanvas.height = 64;
        this._labelCtx = this._labelCanvas.getContext('2d');
    }

    /**
     * Hot-swap the asset provider at runtime.
     * @param {import('./asset_contract.js').AssetProvider} provider
     */
    setAssetProvider(provider) {
        this.assetProvider.dispose();
        this.assetProvider = provider;
        this.clear();
    }

    /**
     * Update NPC state from server tick data.
     * @param {Array} npcData — NPC state objects from server
     */
    updateNPCs(npcData) {
        if (!npcData) return;

        const activeIds = new Set();

        for (const data of npcData) {
            activeIds.add(data.npc_id);
            this._updateMoveState(data);
            if (this.npcMeshes.has(data.npc_id)) {
                this._updateExisting(data);
            } else {
                this._createNPCMesh(data);
            }
        }

        for (const [id, mesh] of this.npcMeshes) {
            if (!activeIds.has(id)) {
                this.npcGroup.remove(mesh.group);
                this.npcMeshes.delete(id);
                this.moveStates.delete(id);
            }
        }
    }

    /**
     * Animate NPCs — called every frame.
     * @param {number} deltaTime — seconds since last frame
     */
    animate(deltaTime) {
        for (const [id, mesh] of this.npcMeshes) {
            const state = this.moveStates.get(id);
            if (!state) continue;

            const group = mesh.group;
            const anim = mesh.animation;

            // --- Walk through trail waypoints ---
            this._walkTrail(state, group, deltaTime);

            // --- Facing direction ---
            if (state.trail.length > 0 && state.trailIndex < state.trail.length) {
                const wp = state.trail[state.trailIndex];
                const dx = wp[0] - state.x;
                const dz = wp[1] - state.z;
                if (Math.abs(dx) > 0.05 || Math.abs(dz) > 0.05) {
                    const targetAngle = Math.atan2(dx, dz);
                    let diff = targetAngle - group.rotation.y;
                    while (diff > Math.PI) diff -= 2 * Math.PI;
                    while (diff < -Math.PI) diff += 2 * Math.PI;
                    group.rotation.y += diff * 0.15;
                }
            }

            // --- Bob animation ---
            const isWalking = state.trail.length > 0 && state.trailIndex < state.trail.length;
            if (mesh.activity === 'idle' || mesh.activity === 'talking') {
                mesh.bobPhase += deltaTime * anim.idle_bob_speed;
                group.position.y = anim.base_y + Math.sin(mesh.bobPhase) * anim.idle_bob_amplitude;
            } else if (mesh.activity === 'walking' || isWalking) {
                mesh.bobPhase += deltaTime * anim.walk_bob_speed;
                group.position.y = anim.base_y + Math.abs(Math.sin(mesh.bobPhase)) * anim.walk_bob_amplitude;
            } else {
                group.position.y = anim.base_y;
            }

            // --- Speech bubble ---
            if (mesh.speechBubble) {
                if (mesh.activity === 'talking') {
                    mesh.speechBubble.visible = true;
                    mesh.speechPulse += deltaTime * 3.0;
                    mesh.speechBubble.scale.setScalar(
                        0.8 + Math.sin(mesh.speechPulse) * 0.1
                    );
                } else {
                    mesh.speechBubble.visible = false;
                }
            }
        }
    }

    clear() {
        for (const mesh of this.npcMeshes.values()) {
            this.npcGroup.remove(mesh.group);
        }
        this.npcMeshes.clear();
        this.moveStates.clear();
    }

    // ---------- Movement ----------

    /**
     * Update movement from server tick.
     *
     * Server sends trail: the exact tiles traversed this tick (max 2-3).
     * Client walks through them in order. When trail is empty, NPC is
     * stationary — snap to server position.
     *
     * No full remaining path. No complex sync. No snap-back possible.
     */
    _updateMoveState(data) {
        const trail = data.trail || [];
        const existing = this.moveStates.get(data.npc_id);

        if (!existing) {
            this.moveStates.set(data.npc_id, {
                x: data.x,
                z: data.z,
                trail: trail,
                trailIndex: 0,
                walkSpeed: 0,
                activity: data.activity,
            });
            return;
        }

        existing.activity = data.activity;

        if (trail.length > 0) {
            // Compute total trail distance for walk speed
            let totalDist = 0;
            let px = existing.x, pz = existing.z;
            for (const wp of trail) {
                const dx = wp[0] - px;
                const dz = wp[1] - pz;
                totalDist += Math.sqrt(dx * dx + dz * dz);
                px = wp[0];
                pz = wp[1];
            }

            if (totalDist > TELEPORT_DISTANCE) {
                // Large jump — teleport to end
                const last = trail[trail.length - 1];
                existing.x = last[0];
                existing.z = last[1];
                existing.trail = [];
                existing.trailIndex = 0;
                existing.walkSpeed = 0;
            } else {
                existing.trail = trail;
                existing.trailIndex = 0;
                // Walk through trail to arrive before next tick
                existing.walkSpeed = totalDist > 0.01
                    ? totalDist / (TICK_INTERVAL * ARRIVAL_FACTOR)
                    : data.move_speed || 2.0;
            }
        } else {
            // NPC didn't move this tick — snap to server position
            existing.trail = [];
            existing.trailIndex = 0;
            existing.walkSpeed = 0;
            existing.x = data.x;
            existing.z = data.z;
        }
    }

    /**
     * Walk through trail waypoints, frame by frame.
     *
     * Each waypoint is an actual tile on the A* path, so movement
     * follows the path exactly — no building clips.
     */
    _walkTrail(state, group, deltaTime) {
        if (!state.trail.length || state.trailIndex >= state.trail.length) {
            // At rest — hold current position
            group.position.x = state.x;
            group.position.z = state.z;
            return;
        }

        let remaining = state.walkSpeed * deltaTime;

        while (remaining > 0 && state.trailIndex < state.trail.length) {
            const wp = state.trail[state.trailIndex];
            const dx = wp[0] - state.x;
            const dz = wp[1] - state.z;
            const dist = Math.sqrt(dx * dx + dz * dz);

            if (dist < 0.01) {
                state.x = wp[0];
                state.z = wp[1];
                state.trailIndex++;
                continue;
            }

            if (remaining >= dist) {
                state.x = wp[0];
                state.z = wp[1];
                remaining -= dist;
                state.trailIndex++;
            } else {
                const frac = remaining / dist;
                state.x += dx * frac;
                state.z += dz * frac;
                remaining = 0;
            }
        }

        group.position.x = state.x;
        group.position.z = state.z;
    }

    // ---------- Mesh creation ----------

    _createNPCMesh(data) {
        const archetype = data.archetype || data.occupation;

        const asset = this.assetProvider.getCharacterAsset(archetype, {
            npc_id: data.npc_id,
            occupation: data.occupation,
            cognition_tier: data.cognition_tier,
        });

        const group = asset.mesh;
        group.name = `npc-${data.npc_id}`;

        const label = this._createNameLabel(data.name);
        const labelPt = asset.attachments.label;
        label.position.set(labelPt.x, labelPt.y, labelPt.z);
        group.add(label);

        const speechBubble = this._createSpeechBubble();
        const speechPt = asset.attachments.speech;
        speechBubble.position.set(speechPt.x, speechPt.y, speechPt.z);
        speechBubble.visible = false;
        group.add(speechBubble);

        const indicatorGeo = new THREE.SphereGeometry(0.04, 6, 4);
        const indicatorMat = new THREE.MeshBasicMaterial({ color: 0x44ff44 });
        const indicator = new THREE.Mesh(indicatorGeo, indicatorMat);
        indicator.position.y = -0.3;
        group.add(indicator);

        group.position.set(data.x, asset.animation.base_y, data.z);
        this.npcGroup.add(group);

        this.npcMeshes.set(data.npc_id, {
            group,
            indicator,
            indicatorMat,
            label,
            speechBubble,
            animation: asset.animation,
            activity: data.activity,
            bobPhase: Math.random() * Math.PI * 2,
            speechPulse: 0,
        });
    }

    _updateExisting(data) {
        const mesh = this.npcMeshes.get(data.npc_id);
        if (!mesh) return;

        mesh.activity = data.activity;

        const activityColours = {
            idle:      0x44ff44,
            walking:   0x4488ff,
            working:   0xff8844,
            sleeping:  0x8844ff,
            talking:   0xffff44,
            eating:    0xff4488,
            gathering: 0x44ffff,
        };
        mesh.indicatorMat.color.setHex(activityColours[data.activity] || 0xffffff);
    }

    _createNameLabel(name) {
        const canvas = this._labelCanvas;
        const ctx = this._labelCtx;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.font = 'bold 24px sans-serif';
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.textAlign = 'center';
        ctx.strokeText(name, canvas.width / 2, canvas.height / 2);
        ctx.fillText(name, canvas.width / 2, canvas.height / 2);

        const labelCanvas = document.createElement('canvas');
        labelCanvas.width = canvas.width;
        labelCanvas.height = canvas.height;
        const labelCtx = labelCanvas.getContext('2d');
        labelCtx.drawImage(canvas, 0, 0);
        const labelTexture = new THREE.CanvasTexture(labelCanvas);

        const spriteMat = new THREE.SpriteMaterial({
            map: labelTexture,
            transparent: true,
            depthTest: false,
        });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(1.2, 0.3, 1);
        return sprite;
    }

    _createSpeechBubble() {
        const geo = new THREE.SphereGeometry(0.08, 6, 4);
        const mat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.9,
        });
        const bubble = new THREE.Mesh(geo, mat);

        const dotGeo = new THREE.SphereGeometry(0.03, 4, 3);
        const dot1 = new THREE.Mesh(dotGeo, mat);
        dot1.position.set(-0.06, -0.06, 0);
        bubble.add(dot1);

        const dot2 = new THREE.Mesh(dotGeo, mat);
        dot2.position.set(-0.03, -0.1, 0);
        bubble.add(dot2);

        return bubble;
    }
}
