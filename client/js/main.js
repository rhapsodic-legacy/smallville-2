/**
 * Smallville 2 — Main client entry point.
 *
 * Initialises Three.js scene, WebSocket connection, and game loop.
 * All game logic is server-side; this is a thin renderer.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { WorldRenderer } from './world_renderer.js';
import { NPCRenderer } from './npc_renderer.js';
import { MemoryInspector } from './memory_inspector.js';

// ---------- Three.js Setup ----------

const canvas = document.getElementById('game-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue
scene.fog = new THREE.Fog(0x87ceeb, 50, 150);

const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    200
);
camera.position.set(20, 25, 20);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI / 2.5; // Don't go below ground

// ---------- Lighting ----------

const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.0);
sunLight.position.set(30, 40, 20);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
sunLight.shadow.camera.left = -40;
sunLight.shadow.camera.right = 40;
sunLight.shadow.camera.top = 40;
sunLight.shadow.camera.bottom = -40;
scene.add(sunLight);

// ---------- World Renderer ----------

const worldRenderer = new WorldRenderer(scene);
const npcRenderer = new NPCRenderer(scene);
let worldLoaded = false;

// ---------- HUD ----------

const hudTime = document.getElementById('hud-time');
const hudGold = document.getElementById('hud-gold');
const statusEl = document.getElementById('hud-status');

function updateHUD(timeData) {
    if (!timeData) return;
    hudTime.textContent = `Day ${timeData.day} — ${timeData.time} (${timeData.phase})`;
}

// ---------- Memory Inspector ----------

const memoryInspector = new MemoryInspector(sendMessage);

// ---------- WebSocket ----------

let ws = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        statusEl.textContent = 'Connected';
        statusEl.style.color = '#44ff44';
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleServerMessage(message);
    };

    ws.onclose = () => {
        statusEl.textContent = 'Disconnected — reconnecting...';
        statusEl.style.color = '#ff4444';
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => {
        statusEl.textContent = 'Connection error';
        statusEl.style.color = '#ff4444';
    };
}

function handleServerMessage(message) {
    switch (message.type) {
        case 'init':
            console.log('Server:', message.message);
            if (message.world) {
                worldRenderer.buildWorld(message.world, message.buildings);
                worldLoaded = true;
            }
            if (message.time) {
                updateHUD(message.time);
                WorldRenderer.updateLighting(sunLight, ambientLight, scene, message.time);
                worldRenderer.updateLamps(message.time.phase);
            }
            if (message.npcs && message.npcs.length > 0) {
                npcRenderer.updateNPCs(message.npcs);
                console.log(`Loaded ${message.npcs.length} NPCs`);
            } else {
                console.warn('WARNING: init message has no NPC data — server may be running stale code');
            }
            break;

        case 'state':
            if (message.world && !worldLoaded) {
                worldRenderer.buildWorld(message.world, message.buildings);
                worldLoaded = true;
            }
            if (message.time) {
                updateHUD(message.time);
                WorldRenderer.updateLighting(sunLight, ambientLight, scene, message.time);
                worldRenderer.updateLamps(message.time.phase);
            }
            if (message.npcs) {
                npcRenderer.updateNPCs(message.npcs);
            }
            break;

        case 'tick':
            if (message.time) {
                updateHUD(message.time);
                WorldRenderer.updateLighting(sunLight, ambientLight, scene, message.time);
                worldRenderer.updateLamps(message.time.phase);
            }
            if (message.npcs) {
                npcRenderer.updateNPCs(message.npcs);
            }
            break;

        case 'chat_response':
            // TODO: Display in chat panel
            break;

        case 'event':
            // TODO: Show notifications
            break;

        case 'pong':
            break;

        default:
            console.warn('Unknown message type:', message.type);
    }
}

function sendMessage(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    }
}

// ---------- Input Handling ----------

document.addEventListener('keydown', (event) => {
    // Memory Inspector toggle
    if (event.code === 'KeyM' && !event.ctrlKey && !event.metaKey) {
        memoryInspector.toggle();
        return;
    }

    const directionMap = {
        'ArrowUp': 'north', 'KeyW': 'north',
        'ArrowDown': 'south', 'KeyS': 'south',
        'ArrowLeft': 'west', 'KeyA': 'west',
        'ArrowRight': 'east', 'KeyD': 'east',
    };

    const direction = directionMap[event.code];
    if (direction) {
        sendMessage({ type: 'move', direction });
    }
});

// ---------- Game Loop ----------

const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    controls.update();
    npcRenderer.animate(delta);
    renderer.render(scene, camera);
}

// ---------- Window Resize ----------

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- Start ----------

connectWebSocket();
animate();
