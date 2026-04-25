/**
 * Player controls — WASD/arrow movement, camera follow, interaction radius.
 *
 * Server-authoritative: key presses send direction to server,
 * server validates and returns updated position each tick.
 * Camera smoothly follows the player position.
 */

import * as THREE from 'three';

const CAMERA_HEIGHT_DEFAULT = 25;
const CAMERA_DISTANCE_DEFAULT = 20;
const CAMERA_LERP_SPEED = 3.0;
const INTERACTION_RADIUS = 3;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 3.0;
const ZOOM_STEP = 0.1;

// Visual indicator for nearby interactable NPCs
const HIGHLIGHT_COLOR = 0xffd700;
const HIGHLIGHT_OPACITY = 0.3;

export class PlayerControls {
    /**
     * @param {THREE.Scene} scene
     * @param {THREE.Camera} camera
     * @param {Function} sendMessage — WebSocket send function
     */
    constructor(scene, camera, sendMessage, npcRenderer = null) {
        this.scene = scene;
        this.camera = camera;
        this.sendMessage = sendMessage;
        // Optional reference to the NPC renderer so the interaction
        // ring can track the avatar's lerped mesh position instead of
        // snapping to the raw server position. Without this, the ring
        // teleports while the avatar smoothly catches up — producing
        // visible drift between ring and avatar during movement.
        this.npcRenderer = npcRenderer;

        // Player state (updated from server ticks)
        this.playerX = 0;
        this.playerZ = 0;
        this.playerMesh = null;
        this.active = false;  // becomes true once server confirms player

        // Camera offset from player (scaled by zoom level)
        this._zoomLevel = 1.0;
        this.cameraOffset = new THREE.Vector3(
            CAMERA_DISTANCE_DEFAULT, CAMERA_HEIGHT_DEFAULT, CAMERA_DISTANCE_DEFAULT
        );
        this.cameraTarget = new THREE.Vector3();
        this.cameraFollowing = true;

        // Keys currently held
        this._keysHeld = new Set();
        this._chatOpen = false;

        // Interaction radius indicator (subtle ground ring)
        this._radiusIndicator = this._createRadiusIndicator();
        scene.add(this._radiusIndicator);

        // Nearby NPC highlights
        this._nearbyNpcIds = new Set();

        // Bind input handlers
        this._onKeyDown = this._onKeyDown.bind(this);
        this._onKeyUp = this._onKeyUp.bind(this);
        this._onWheel = this._onWheel.bind(this);
        document.addEventListener('keydown', this._onKeyDown);
        document.addEventListener('keyup', this._onKeyUp);
        document.addEventListener('wheel', this._onWheel, { passive: false });
    }

    /** Activate player mode with initial position. */
    activate(x, z) {
        this.playerX = x;
        this.playerZ = z;
        this.active = true;

        // Snap radius indicator to player (+0.5 for tile centre)
        this._radiusIndicator.position.set(x + 0.5, 0.05, z + 0.5);

        // Snap camera to player immediately
        this.cameraTarget.set(x, 0, z);
        this.camera.position.set(
            x + this.cameraOffset.x,
            this.cameraOffset.y,
            z + this.cameraOffset.z
        );
        this.camera.lookAt(this.cameraTarget);
    }

    /** Set whether chat input is focused (disables movement keys). */
    setChatOpen(open) {
        this._chatOpen = open;
    }

    /** Update from server tick — authoritative player position. */
    updateFromServer(playerData) {
        if (!playerData) return;
        this.playerX = playerData.x;
        this.playerZ = playerData.z;
        // Ring position is updated every frame in update() to track the
        // lerped avatar mesh — don't snap it here or it drifts visibly.
    }

    /** Update nearby NPC IDs for highlighting. */
    updateNearbyNpcs(npcs) {
        if (!this.active || !npcs) return;
        const newNearby = new Set();
        for (const npc of npcs) {
            // Filter out the player NPC by id (plain NPC.to_dict lacks is_player)
            if (npc.npc_id === 'player' || npc.is_player) continue;
            const dist = Math.abs(npc.x - this.playerX) + Math.abs(npc.z - this.playerZ);
            if (dist <= INTERACTION_RADIUS) {
                newNearby.add(npc.npc_id);
            }
        }
        this._nearbyNpcIds = newNearby;
    }

    /** Returns set of NPC IDs within interaction radius. */
    getNearbyNpcIds() {
        return this._nearbyNpcIds;
    }

    /** Per-frame update: send movement, smooth camera. */
    update(deltaTime) {
        if (!this.active) return;

        // Send movement direction if keys held and chat not open
        if (!this._chatOpen) {
            const dir = this._getDirection();
            if (dir) {
                this.sendMessage({ type: 'player_move', direction: dir });
            }
        }

        // Track the avatar's rendered position so the ring/camera follow
        // the visible mesh, not the (discrete) server position. Falls
        // back to server position if the avatar mesh isn't yet created.
        const avatar = this._getAvatarMesh();
        const followX = avatar ? avatar.position.x - 0.5 : this.playerX;
        const followZ = avatar ? avatar.position.z - 0.5 : this.playerZ;

        // Keep the interaction ring centred on the avatar every frame
        // (+0.5 aligns with the tile-centre convention the renderer uses).
        this._radiusIndicator.position.set(followX + 0.5, 0.05, followZ + 0.5);

        // Smooth camera follow
        if (this.cameraFollowing) {
            this.cameraTarget.lerp(
                new THREE.Vector3(followX, 0, followZ),
                CAMERA_LERP_SPEED * deltaTime
            );
            this.camera.position.set(
                this.cameraTarget.x + this.cameraOffset.x,
                this.cameraOffset.y,
                this.cameraTarget.z + this.cameraOffset.z
            );
            this.camera.lookAt(this.cameraTarget);
        }
    }

    /** Return the player's avatar mesh group if the renderer knows it. */
    _getAvatarMesh() {
        if (!this.npcRenderer || !this.npcRenderer.npcMeshes) return null;
        const entry = this.npcRenderer.npcMeshes.get('player');
        return entry ? entry.group : null;
    }

    /** Toggle camera between follow mode and free orbit. */
    toggleCameraMode() {
        this.cameraFollowing = !this.cameraFollowing;
        return this.cameraFollowing;
    }

    /** Clean up event listeners. */
    dispose() {
        document.removeEventListener('keydown', this._onKeyDown);
        document.removeEventListener('keyup', this._onKeyUp);
        document.removeEventListener('wheel', this._onWheel);
        if (this._radiusIndicator) {
            this.scene.remove(this._radiusIndicator);
        }
    }

    // --- Private ---

    _onKeyDown(event) {
        // Don't capture input when typing in chat
        if (this._chatOpen) return;
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;

        const moveKeys = ['KeyW', 'KeyA', 'KeyS', 'KeyD',
                          'ArrowUp', 'ArrowLeft', 'ArrowDown', 'ArrowRight'];
        if (moveKeys.includes(event.code)) {
            this._keysHeld.add(event.code);
            event.preventDefault();
        }

        // Tab toggles camera mode
        if (event.code === 'Tab') {
            event.preventDefault();
            this.toggleCameraMode();
        }
    }

    _onKeyUp(event) {
        this._keysHeld.delete(event.code);
    }

    _onWheel(event) {
        if (!this.active || !this.cameraFollowing) return;
        event.preventDefault();
        const delta = event.deltaY > 0 ? ZOOM_STEP : -ZOOM_STEP;
        this._zoomLevel = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this._zoomLevel + delta));
        this.cameraOffset.set(
            CAMERA_DISTANCE_DEFAULT * this._zoomLevel,
            CAMERA_HEIGHT_DEFAULT * this._zoomLevel,
            CAMERA_DISTANCE_DEFAULT * this._zoomLevel,
        );
    }

    _getDirection() {
        if (this._keysHeld.has('KeyW') || this._keysHeld.has('ArrowUp')) return 'north';
        if (this._keysHeld.has('KeyS') || this._keysHeld.has('ArrowDown')) return 'south';
        if (this._keysHeld.has('KeyA') || this._keysHeld.has('ArrowLeft')) return 'west';
        if (this._keysHeld.has('KeyD') || this._keysHeld.has('ArrowRight')) return 'east';
        return null;
    }

    _createRadiusIndicator() {
        const geometry = new THREE.RingGeometry(
            INTERACTION_RADIUS - 0.1,
            INTERACTION_RADIUS + 0.1,
            32
        );
        geometry.rotateX(-Math.PI / 2);
        const material = new THREE.MeshBasicMaterial({
            color: HIGHLIGHT_COLOR,
            transparent: true,
            opacity: HIGHLIGHT_OPACITY,
            side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.y = 0.05;
        return mesh;
    }
}
