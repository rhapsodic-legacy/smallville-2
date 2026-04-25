/**
 * Smallville 2 — Main client entry point.
 *
 * Initialises Three.js scene, WebSocket connection, player controls,
 * chat, trade, HUD, and game loop. All game logic is server-side;
 * this is a thin renderer + input handler.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { WorldRenderer } from './world_renderer.js';
import { NPCRenderer } from './npc_renderer.js';
import { MemoryInspector } from './memory_inspector.js';
import { PlayerControls } from './player_controls.js';
import { ChatUI } from './chat_ui.js';
import { TradeUI } from './trade_ui.js';
import { HUD } from './hud.js';

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
let buildingsCache = null;

// ---------- HUD ----------

const hud = new HUD();

// ---------- Player Controls ----------

const playerControls = new PlayerControls(scene, camera, sendMessage, npcRenderer);

// ---------- Chat UI ----------

const chatUI = new ChatUI(sendMessage, (isOpen) => {
    playerControls.setChatOpen(isOpen);
});

// ---------- Trade UI ----------

const tradeUI = new TradeUI(sendMessage, (isOpen) => {
    playerControls.setChatOpen(isOpen);  // Reuse — blocks movement while trading
});

// ---------- Memory Inspector ----------

const memoryInspector = new MemoryInspector(sendMessage);

// ---------- WebSocket ----------

let ws = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        hud.setStatus('Connected', '#44ff44');
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleServerMessage(message);
        } catch (err) {
            console.error('Message handler error:', err);
        }
    };

    ws.onclose = () => {
        hud.setStatus('Disconnected — reconnecting...', '#ff4444');
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => {
        hud.setStatus('Connection error', '#ff4444');
    };
}

function handleServerMessage(message) {
    switch (message.type) {
        case 'init':
            console.log('Server:', message.message);
            if (message.world) {
                worldRenderer.buildWorld(message.world, message.buildings);
                worldLoaded = true;
                buildingsCache = message.buildings;
                if (message.world.width) {
                    hud.setWorldSize(message.world.width, message.world.height);
                }
            }
            if (message.time) {
                hud.updateTime(message.time);
                WorldRenderer.updateLighting(sunLight, ambientLight, scene, message.time);
                worldRenderer.updateLamps(message.time.phase);
            }
            if (message.npcs && message.npcs.length > 0) {
                npcRenderer.updateNPCs(message.npcs);
                console.log(`Loaded ${message.npcs.length} NPCs`);
            }
            // Activate player if present
            if (message.player) {
                playerControls.activate(message.player.x, message.player.z);
                hud.updatePlayerStats(message.player);
                hud.notify('Welcome to Smallville! Use WASD to move.', 'info');

                // Disable orbit controls — PlayerControls handles zoom via scroll
                controls.enabled = false;
            }
            break;

        case 'state':
            if (message.world && !worldLoaded) {
                worldRenderer.buildWorld(message.world, message.buildings);
                worldLoaded = true;
                buildingsCache = message.buildings;
            }
            if (message.time) {
                hud.updateTime(message.time);
                WorldRenderer.updateLighting(sunLight, ambientLight, scene, message.time);
                worldRenderer.updateLamps(message.time.phase);
            }
            if (message.npcs) {
                npcRenderer.updateNPCs(message.npcs);
            }
            break;

        case 'tick':
            if (message.time) {
                hud.updateTime(message.time);
                WorldRenderer.updateLighting(sunLight, ambientLight, scene, message.time);
                worldRenderer.updateLamps(message.time.phase);
            }
            if (message.npcs) {
                npcRenderer.updateNPCs(message.npcs);
                playerControls.updateNearbyNpcs(message.npcs);

                // Update nearby NPC list for chat/trade
                const nearbyIds = playerControls.getNearbyNpcIds();
                const nearbyNpcs = message.npcs
                    .filter(n => nearbyIds.has(n.npc_id))
                    .sort((a, b) => {
                        const da = Math.abs(a.x - playerControls.playerX) + Math.abs(a.z - playerControls.playerZ);
                        const db = Math.abs(b.x - playerControls.playerX) + Math.abs(b.z - playerControls.playerZ);
                        return da - db;
                    });
                chatUI.setNearbyNpcs(nearbyNpcs);
                tradeUI.setNearbyNpcs(nearbyNpcs);
                hud.updateNearbyNpcs(nearbyNpcs);
            }
            if (message.player) {
                playerControls.updateFromServer(message.player);
                hud.updatePlayerStats(message.player);
                tradeUI.updatePlayerState(message.player);
                hud.updateMinimap(message.npcs, message.player, buildingsCache);
            }
            if (message.conversations) {
                chatUI.handleConversationUpdate(message.conversations);
            }
            if (message.town_agenda) {
                hud.updateTownAgenda(message.town_agenda);
            }
            if (message.memory_events) {
                for (const evt of message.memory_events) {
                    npcRenderer.flashMemory(
                        evt.npc_id, evt.importance, evt.category,
                    );
                    hud.recordMemoryEvent(evt);
                }
            }
            break;

        case 'chat_response':
            chatUI.handleResponse(message);
            break;

        case 'trade_response':
            tradeUI.handleResponse(message);
            break;

        case 'event':
            if (message.events) {
                for (const evt of message.events) {
                    hud.notify(evt.description || evt.type, 'event');
                }
            }
            break;

        case 'pong':
            break;

        default:
            // Memory inspector and other handlers
            if (message.type === 'memory_data' || message.type === 'memory_stats') {
                // Memory inspector handles these internally
            } else {
                console.warn('Unknown message type:', message.type);
            }
    }
}

function sendMessage(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    }
}

// ---------- Input Handling ----------

document.addEventListener('keydown', (event) => {
    // Don't capture when typing
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    // Don't steal key events while the thinking-level <select> is focused.
    if (event.target.tagName === 'SELECT') return;

    // Memory Inspector toggle
    if (event.code === 'KeyM' && !event.ctrlKey && !event.metaKey) {
        memoryInspector.toggle();
        return;
    }
});

// ---------- Thinking-level toggle ----------

const thinkingSelect = document.getElementById('hud-thinking-level');
if (thinkingSelect) {
    // Persist choice across page reloads.
    const saved = localStorage.getItem('smallville.thinking_level');
    if (saved && ['fast', 'balanced', 'deep'].includes(saved)) {
        thinkingSelect.value = saved;
    }
    const sendLevel = (lvl) => sendMessage({ type: 'set_thinking_level', level: lvl });

    thinkingSelect.addEventListener('change', (e) => {
        const level = e.target.value;
        localStorage.setItem('smallville.thinking_level', level);
        sendLevel(level);
        hud.notify(`NPC thinking: ${level}`, 'info');
    });

    // Push the saved preference to the server once the socket opens.
    const origOnOpen = () => sendLevel(thinkingSelect.value);
    // connectWebSocket installs ws.onopen itself; attach a one-shot sync
    // after the initial connect.
    const syncWhenOpen = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            origOnOpen();
        } else {
            setTimeout(syncWhenOpen, 200);
        }
    };
    setTimeout(syncWhenOpen, 500);
}

// ---------- Game Loop ----------

const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    try {
        const delta = clock.getDelta();

        // Player controls update (sends movement, smooths camera)
        playerControls.update(delta);

        // Orbit controls only if player not following
        if (!playerControls.cameraFollowing || !playerControls.active) {
            controls.update();
        }

        npcRenderer.animate(delta);
        renderer.render(scene, camera);
    } catch (err) {
        console.error('Animate loop error:', err);
    }
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
