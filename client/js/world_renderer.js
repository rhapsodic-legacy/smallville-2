/**
 * Smallville 2 — World Renderer
 *
 * Converts server-sent grid state into Three.js geometry.
 * Handles terrain meshes, building meshes, resource nodes,
 * camera controls, and day/night lighting.
 */

import * as THREE from 'three';

// ---------- Terrain colours ----------

const TERRAIN_COLOURS = {
    grass:  0x4a7c2e,
    dirt:   0x8b7355,
    road:   0x9e9e7a,
    water:  0x3366aa,
    stone:  0x808080,
    sand:   0xc2b280,
    forest: 0x2d5a1e,
};

// ---------- Building colours ----------

const BUILDING_COLOURS = {
    tavern:       0x8b4513,
    blacksmith:   0x555555,
    market_stall: 0xb8860b,
    church:       0xd4c4a8,
    town_hall:    0x6b4226,
    home:         0xa0522d,
    farm:         0x7a6032,
};

const ROOF_COLOURS = {
    tavern:       0x8b0000,
    blacksmith:   0x333333,
    market_stall: 0xdaa520,
    church:       0x666666,
    town_hall:    0x4a2c0a,
    home:         0x8b3a3a,
    farm:         0x556b2f,
};

// ---------- WorldRenderer class ----------

export class WorldRenderer {
    constructor(scene) {
        this.scene = scene;
        this.terrainGroup = new THREE.Group();
        this.buildingGroup = new THREE.Group();
        this.resourceGroup = new THREE.Group();
        this.featureGroup = new THREE.Group();
        this.lampGroup = new THREE.Group();
        this.scene.add(this.terrainGroup);
        this.scene.add(this.buildingGroup);
        this.scene.add(this.resourceGroup);
        this.scene.add(this.featureGroup);
        this.scene.add(this.lampGroup);

        /** @type {THREE.PointLight[]} */
        this.lampLights = [];

        // Reusable geometry for terrain tiles (instanced later)
        this._tileGeo = new THREE.PlaneGeometry(1, 1);
        this._tileGeo.rotateX(-Math.PI / 2);
    }

    /**
     * Rebuild the entire world from server state.
     * @param {Object} worldData — { width, height, tiles: [...] }
     * @param {Array} buildings — optional building list with door_x, door_z
     */
    buildWorld(worldData, buildings) {
        this.clearWorld();

        if (!worldData || !worldData.tiles) return;

        this._buildTerrain(worldData.tiles);
        this._buildStructures(worldData.tiles);
        this._buildResources(worldData.tiles);
        this._buildFeatures(worldData.tiles);
        this._buildLampPosts(worldData.tiles);
        if (buildings) this._buildDoorMarkers(buildings);
    }

    clearWorld() {
        this._clearGroup(this.terrainGroup);
        this._clearGroup(this.buildingGroup);
        this._clearGroup(this.resourceGroup);
        this._clearGroup(this.featureGroup);
        this._clearGroup(this.lampGroup);
        this.lampLights = [];
    }

    // ---------- Terrain ----------

    _buildTerrain(tiles) {
        // Group tiles by terrain type for batched rendering
        const byTerrain = {};
        for (const tile of tiles) {
            const key = tile.terrain;
            if (!byTerrain[key]) byTerrain[key] = [];
            byTerrain[key].push(tile);
        }

        for (const [terrain, group] of Object.entries(byTerrain)) {
            const colour = TERRAIN_COLOURS[terrain] || 0x888888;
            const mesh = this._createTerrainBatch(group, colour, terrain);
            this.terrainGroup.add(mesh);
        }
    }

    _createTerrainBatch(tiles, colour, terrain) {
        const geo = new THREE.PlaneGeometry(1, 1);
        geo.rotateX(-Math.PI / 2);

        const mat = new THREE.MeshStandardMaterial({
            color: colour,
            roughness: terrain === 'water' ? 0.2 : 0.9,
            metalness: terrain === 'water' ? 0.3 : 0.0,
        });

        const instanced = new THREE.InstancedMesh(geo, mat, tiles.length);
        const matrix = new THREE.Matrix4();

        for (let i = 0; i < tiles.length; i++) {
            const t = tiles[i];
            const y = (t.elevation || 0) * 0.3;
            matrix.setPosition(t.x + 0.5, y, t.z + 0.5);
            instanced.setMatrixAt(i, matrix);
        }

        instanced.receiveShadow = true;
        return instanced;
    }

    // ---------- Buildings ----------

    _buildStructures(tiles) {
        // Find tiles with building objects — only the anchor tile (first object)
        const seen = new Set();
        for (const tile of tiles) {
            if (!tile.objects) continue;
            for (const obj of tile.objects) {
                if (obj.object_type !== 'building') continue;
                if (seen.has(obj.object_id)) continue;
                seen.add(obj.object_id);
                this._createBuilding(obj, tile);
            }
        }
    }

    _createBuilding(obj, anchorTile) {
        const bType = obj.object_id.replace(/_\d+$/, '');
        const wallColour = BUILDING_COLOURS[bType] || 0x8b4513;
        const roofColour = ROOF_COLOURS[bType] || 0x8b0000;

        // Use metadata for dimensions, fallback to 2x2
        const w = (obj.metadata && obj.metadata.width) || 2;
        const h = (obj.metadata && obj.metadata.height) || 2;
        const wallHeight = bType === 'church' ? 3.5 : 2.0;
        const roofHeight = bType === 'church' ? 2.0 : 1.0;
        const wallThick = 0.15;

        // Door position relative to building origin (0,0 = NW corner)
        const doorLocalX = (obj.metadata && obj.metadata.door_local_x != null)
            ? obj.metadata.door_local_x
            : Math.floor(w / 2);

        const group = new THREE.Group();
        const wallMat = new THREE.MeshStandardMaterial({
            color: wallColour,
            roughness: 0.8,
        });

        // --- 4 walls with door gap on south face ---

        // North wall (full width)
        const northGeo = new THREE.BoxGeometry(w, wallHeight, wallThick);
        const north = new THREE.Mesh(northGeo, wallMat);
        north.position.set(0, wallHeight / 2, -(h / 2) + wallThick / 2);
        north.castShadow = true;
        north.receiveShadow = true;
        group.add(north);

        // East wall (full depth)
        const eastGeo = new THREE.BoxGeometry(wallThick, wallHeight, h);
        const east = new THREE.Mesh(eastGeo, wallMat);
        east.position.set(w / 2 - wallThick / 2, wallHeight / 2, 0);
        east.castShadow = true;
        east.receiveShadow = true;
        group.add(east);

        // West wall (full depth)
        const west = new THREE.Mesh(eastGeo, wallMat);
        west.position.set(-(w / 2) + wallThick / 2, wallHeight / 2, 0);
        west.castShadow = true;
        west.receiveShadow = true;
        group.add(west);

        // South wall — split into left and right segments around door gap
        const doorWidth = 0.7;
        const doorCentreX = doorLocalX - (w / 2) + 0.5;  // local coords relative to group centre

        const southZ = h / 2 - wallThick / 2;

        // Left segment (from west edge to door gap)
        const leftLen = (doorCentreX + w / 2) - doorWidth / 2;
        if (leftLen > 0.05) {
            const leftGeo = new THREE.BoxGeometry(leftLen, wallHeight, wallThick);
            const leftWall = new THREE.Mesh(leftGeo, wallMat);
            leftWall.position.set(-(w / 2) + leftLen / 2, wallHeight / 2, southZ);
            leftWall.castShadow = true;
            leftWall.receiveShadow = true;
            group.add(leftWall);
        }

        // Right segment (from door gap to east edge)
        const rightStart = (doorCentreX + w / 2) + doorWidth / 2;
        const rightLen = w - rightStart;
        if (rightLen > 0.05) {
            const rightGeo = new THREE.BoxGeometry(rightLen, wallHeight, wallThick);
            const rightWall = new THREE.Mesh(rightGeo, wallMat);
            rightWall.position.set(w / 2 - rightLen / 2, wallHeight / 2, southZ);
            rightWall.castShadow = true;
            rightWall.receiveShadow = true;
            group.add(rightWall);
        }

        // Floor
        const floorGeo = new THREE.BoxGeometry(w - wallThick * 2, 0.05, h - wallThick * 2);
        const floorMat = new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.9,
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.position.y = 0.025;
        floor.receiveShadow = true;
        group.add(floor);

        // Roof
        if (bType === 'church') {
            const steepleGeo = new THREE.ConeGeometry(1.0, roofHeight * 2, 4);
            const steebleMat = new THREE.MeshStandardMaterial({
                color: roofColour,
                roughness: 0.6,
            });
            const steeple = new THREE.Mesh(steepleGeo, steebleMat);
            steeple.position.y = wallHeight + roofHeight;
            steeple.castShadow = true;
            group.add(steeple);
        } else if (bType === 'market_stall') {
            const canopyGeo = new THREE.BoxGeometry(w + 0.4, 0.15, h + 0.4);
            const canopyMat = new THREE.MeshStandardMaterial({
                color: roofColour,
                roughness: 0.5,
            });
            const canopy = new THREE.Mesh(canopyGeo, canopyMat);
            canopy.position.y = wallHeight + 0.1;
            canopy.castShadow = true;
            group.add(canopy);
        } else {
            const roofGeo = new THREE.ConeGeometry(
                Math.max(w, h) * 0.7, roofHeight, 4
            );
            const roofMat = new THREE.MeshStandardMaterial({
                color: roofColour,
                roughness: 0.6,
            });
            const roof = new THREE.Mesh(roofGeo, roofMat);
            roof.position.y = wallHeight + roofHeight / 2;
            roof.rotation.y = Math.PI / 4;
            roof.castShadow = true;
            group.add(roof);
        }

        // Position at centre of footprint
        group.position.set(
            anchorTile.x + w / 2,
            (anchorTile.elevation || 0) * 0.3,
            anchorTile.z + h / 2,
        );

        this.buildingGroup.add(group);
    }

    // ---------- Resources ----------

    _buildResources(tiles) {
        for (const tile of tiles) {
            if (!tile.objects) continue;
            for (const obj of tile.objects) {
                if (obj.object_type !== 'resource') continue;
                this._createResource(obj, tile);
            }
        }
    }

    _createResource(obj, tile) {
        const group = new THREE.Group();
        const name = obj.name.toLowerCase();

        if (name.includes('tree') || name.includes('oak')) {
            // Trunk
            const trunkGeo = new THREE.CylinderGeometry(0.1, 0.15, 1.0, 6);
            const trunkMat = new THREE.MeshStandardMaterial({ color: 0x5c3a1e });
            const trunk = new THREE.Mesh(trunkGeo, trunkMat);
            trunk.position.y = 0.5;
            trunk.castShadow = true;
            group.add(trunk);

            // Canopy
            const canopyGeo = new THREE.SphereGeometry(0.5, 6, 6);
            const canopyMat = new THREE.MeshStandardMaterial({ color: 0x2d8a4e });
            const canopy = new THREE.Mesh(canopyGeo, canopyMat);
            canopy.position.y = 1.3;
            canopy.castShadow = true;
            group.add(canopy);
        } else if (name.includes('bush') || name.includes('berry')) {
            const bushGeo = new THREE.SphereGeometry(0.3, 5, 5);
            const bushMat = new THREE.MeshStandardMaterial({ color: 0x3a7a3a });
            const bush = new THREE.Mesh(bushGeo, bushMat);
            bush.position.y = 0.3;
            bush.castShadow = true;
            group.add(bush);
        } else if (name.includes('iron') || name.includes('stone') || name.includes('quarry')) {
            const rockGeo = new THREE.DodecahedronGeometry(0.3, 0);
            const rockMat = new THREE.MeshStandardMaterial({
                color: 0x777788,
                roughness: 0.9,
            });
            const rock = new THREE.Mesh(rockGeo, rockMat);
            rock.position.y = 0.3;
            rock.castShadow = true;
            group.add(rock);
        } else {
            // Generic marker
            const markerGeo = new THREE.BoxGeometry(0.4, 0.4, 0.4);
            const markerMat = new THREE.MeshStandardMaterial({ color: 0xaaaa44 });
            const marker = new THREE.Mesh(markerGeo, markerMat);
            marker.position.y = 0.2;
            group.add(marker);
        }

        group.position.set(
            tile.x + 0.5,
            (tile.elevation || 0) * 0.3,
            tile.z + 0.5,
        );
        this.resourceGroup.add(group);
    }

    // ---------- Terrain Features (structures & decorations) ----------

    _buildFeatures(tiles) {
        const seen = new Set();
        for (const tile of tiles) {
            if (!tile.objects) continue;
            for (const obj of tile.objects) {
                if (obj.object_type !== 'structure' && obj.object_type !== 'decoration') continue;
                if (seen.has(obj.object_id)) continue;
                seen.add(obj.object_id);
                this._createFeature(obj, tile);
            }
        }
    }

    _createFeature(obj, tile) {
        const feature = (obj.metadata && obj.metadata.feature) || '';
        const group = new THREE.Group();
        const y = (tile.elevation || 0) * 0.3;

        switch (feature) {
            case 'bridge':
                this._buildBridge(group);
                break;
            case 'well':
                this._buildWell(group);
                break;
            case 'campfire':
                this._buildCampfire(group);
                break;
            case 'garden':
                this._buildGardenPlant(group, obj.name);
                break;
            case 'watchtower':
                // Watchtowers use object_type 'building', rendered by _buildStructures
                return;
            default:
                // Generic marker for unknown features
                this._buildFeatureMarker(group, feature);
                break;
        }

        group.position.set(tile.x + 0.5, y, tile.z + 0.5);
        this.featureGroup.add(group);
    }

    _buildBridge(group) {
        // Wooden bridge planks
        const plankMat = new THREE.MeshStandardMaterial({
            color: 0x8b6914,
            roughness: 0.85,
        });

        // Main deck
        const deckGeo = new THREE.BoxGeometry(1.0, 0.08, 1.0);
        const deck = new THREE.Mesh(deckGeo, plankMat);
        deck.position.y = 0.15;
        deck.receiveShadow = true;
        group.add(deck);

        // Railings
        const railMat = new THREE.MeshStandardMaterial({
            color: 0x6b4f12,
            roughness: 0.8,
        });
        for (const side of [-0.45, 0.45]) {
            // Vertical posts
            for (const along of [-0.35, 0.35]) {
                const postGeo = new THREE.CylinderGeometry(0.025, 0.025, 0.5, 4);
                const post = new THREE.Mesh(postGeo, railMat);
                post.position.set(side, 0.44, along);
                post.castShadow = true;
                group.add(post);
            }
            // Horizontal rail
            const railGeo = new THREE.BoxGeometry(0.03, 0.03, 0.8);
            const rail = new THREE.Mesh(railGeo, railMat);
            rail.position.set(side, 0.65, 0);
            group.add(rail);
        }
    }

    _buildWell(group) {
        const stoneMat = new THREE.MeshStandardMaterial({
            color: 0x808080,
            roughness: 0.9,
        });

        // Circular stone wall
        const wallGeo = new THREE.CylinderGeometry(0.3, 0.35, 0.5, 8, 1, true);
        const wall = new THREE.Mesh(wallGeo, stoneMat);
        wall.position.y = 0.25;
        wall.castShadow = true;
        group.add(wall);

        // Rim
        const rimGeo = new THREE.TorusGeometry(0.32, 0.04, 6, 8);
        const rim = new THREE.Mesh(rimGeo, stoneMat);
        rim.position.y = 0.5;
        rim.rotation.x = -Math.PI / 2;
        group.add(rim);

        // Wooden support posts
        const woodMat = new THREE.MeshStandardMaterial({ color: 0x5c3a1e });
        for (const side of [-0.25, 0.25]) {
            const postGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.8, 4);
            const post = new THREE.Mesh(postGeo, woodMat);
            post.position.set(side, 0.9, 0);
            post.castShadow = true;
            group.add(post);
        }

        // Crossbar
        const barGeo = new THREE.CylinderGeometry(0.025, 0.025, 0.6, 4);
        const bar = new THREE.Mesh(barGeo, woodMat);
        bar.position.y = 1.3;
        bar.rotation.z = Math.PI / 2;
        group.add(bar);

        // Bucket (tiny cylinder hanging from crossbar)
        const bucketGeo = new THREE.CylinderGeometry(0.06, 0.05, 0.1, 6);
        const bucketMat = new THREE.MeshStandardMaterial({ color: 0x4a4a4a });
        const bucket = new THREE.Mesh(bucketGeo, bucketMat);
        bucket.position.y = 1.0;
        group.add(bucket);
    }

    _buildCampfire(group) {
        // Stone ring
        const stoneMat = new THREE.MeshStandardMaterial({
            color: 0x666666,
            roughness: 0.95,
        });
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const stoneGeo = new THREE.SphereGeometry(0.06, 4, 4);
            const stone = new THREE.Mesh(stoneGeo, stoneMat);
            stone.position.set(
                Math.cos(angle) * 0.22,
                0.05,
                Math.sin(angle) * 0.22,
            );
            group.add(stone);
        }

        // Logs
        const logMat = new THREE.MeshStandardMaterial({ color: 0x3a2010 });
        for (let i = 0; i < 3; i++) {
            const logGeo = new THREE.CylinderGeometry(0.03, 0.035, 0.3, 5);
            const log = new THREE.Mesh(logGeo, logMat);
            const angle = (i / 3) * Math.PI * 2;
            log.position.set(Math.cos(angle) * 0.08, 0.08, Math.sin(angle) * 0.08);
            log.rotation.z = Math.PI / 2 + angle;
            group.add(log);
        }

        // Ember glow (emissive sphere)
        const emberGeo = new THREE.SphereGeometry(0.08, 6, 6);
        const emberMat = new THREE.MeshStandardMaterial({
            color: 0xff4400,
            emissive: 0xff4400,
            emissiveIntensity: 0.6,
            roughness: 1.0,
        });
        const ember = new THREE.Mesh(emberGeo, emberMat);
        ember.position.y = 0.12;
        group.add(ember);

        // Warm point light
        const light = new THREE.PointLight(0xff6622, 0.5, 5, 2);
        light.position.y = 0.3;
        group.add(light);
    }

    _buildGardenPlant(group, name) {
        const n = (name || '').toLowerCase();
        let colour = 0x88cc44;
        let height = 0.25;

        if (n.includes('rose')) {
            colour = 0xcc3344;
            height = 0.35;
        } else if (n.includes('lavender')) {
            colour = 0x9966cc;
            height = 0.3;
        } else if (n.includes('herb')) {
            colour = 0x55aa33;
            height = 0.2;
        } else if (n.includes('wildflower')) {
            colour = 0xddaa33;
            height = 0.3;
        }

        // Low bush
        const bushGeo = new THREE.SphereGeometry(0.15, 5, 4);
        const bushMat = new THREE.MeshStandardMaterial({
            color: 0x44882a,
            roughness: 0.9,
        });
        const bush = new THREE.Mesh(bushGeo, bushMat);
        bush.position.y = 0.12;
        group.add(bush);

        // Flower/herb on top
        const flowerGeo = new THREE.SphereGeometry(0.08, 5, 4);
        const flowerMat = new THREE.MeshStandardMaterial({
            color: colour,
            roughness: 0.7,
        });
        const flower = new THREE.Mesh(flowerGeo, flowerMat);
        flower.position.y = height;
        group.add(flower);
    }

    _buildFeatureMarker(group, feature) {
        // Subtle stone marker for unrecognised features
        const geo = new THREE.DodecahedronGeometry(0.2, 0);
        const mat = new THREE.MeshStandardMaterial({
            color: 0x999999,
            roughness: 0.9,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.y = 0.2;
        mesh.castShadow = true;
        group.add(mesh);
    }

    // ---------- Victorian Lamp Posts ----------

    _buildLampPosts(tiles) {
        // Place lamp posts along road tiles at regular intervals
        const roadTiles = tiles.filter(t => t.terrain === 'road');
        const SPACING = 6;  // one lamp every ~6 road tiles

        // Sort road tiles for consistent placement
        roadTiles.sort((a, b) => a.x !== b.x ? a.x - b.x : a.z - b.z);

        const placed = new Set();
        for (let i = 0; i < roadTiles.length; i += SPACING) {
            const t = roadTiles[i];
            const key = `${t.x},${t.z}`;
            if (placed.has(key)) continue;
            placed.add(key);

            const y = (t.elevation || 0) * 0.3;
            this._createLampPost(t.x + 0.5, y, t.z + 0.5);
        }
    }

    _createLampPost(x, baseY, z) {
        const group = new THREE.Group();
        const ironColour = 0x1a1a1a;
        const ironMat = new THREE.MeshStandardMaterial({
            color: ironColour,
            roughness: 0.4,
            metalness: 0.8,
        });

        // Base — ornate square pedestal
        const baseGeo = new THREE.BoxGeometry(0.25, 0.12, 0.25);
        const base = new THREE.Mesh(baseGeo, ironMat);
        base.position.y = 0.06;
        group.add(base);

        // Main pole — tapered cylinder
        const poleGeo = new THREE.CylinderGeometry(0.04, 0.06, 1.8, 8);
        const pole = new THREE.Mesh(poleGeo, ironMat);
        pole.position.y = 1.02;
        pole.castShadow = true;
        group.add(pole);

        // Decorative collar at middle
        const collarGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.06, 8);
        const collar = new THREE.Mesh(collarGeo, ironMat);
        collar.position.y = 0.9;
        group.add(collar);

        // Lantern housing — four-sided Victorian cage
        const housingGeo = new THREE.CylinderGeometry(0.1, 0.12, 0.25, 4);
        const glassMat = new THREE.MeshStandardMaterial({
            color: 0xffeeaa,
            roughness: 0.1,
            metalness: 0.0,
            transparent: true,
            opacity: 0.6,
            emissive: 0xffaa44,
            emissiveIntensity: 0.3,
        });
        const housing = new THREE.Mesh(housingGeo, glassMat);
        housing.position.y = 2.05;
        housing.rotation.y = Math.PI / 4;
        group.add(housing);

        // Lantern cap — small cone
        const capGeo = new THREE.ConeGeometry(0.13, 0.1, 4);
        const cap = new THREE.Mesh(capGeo, ironMat);
        cap.position.y = 2.22;
        cap.rotation.y = Math.PI / 4;
        group.add(cap);

        // Finial — tiny sphere on top
        const finialGeo = new THREE.SphereGeometry(0.03, 6, 4);
        const finial = new THREE.Mesh(finialGeo, ironMat);
        finial.position.y = 2.3;
        group.add(finial);

        // Point light — warm glow
        const light = new THREE.PointLight(0xffaa44, 0, 8, 2);
        light.position.y = 2.05;
        group.add(light);
        this.lampLights.push(light);

        group.position.set(x, baseY, z);
        this.lampGroup.add(group);
    }

    /**
     * Update lamp post lights based on time of day.
     * Call this from the main loop alongside updateLighting.
     * @param {string} phase — 'dawn', 'day', 'dusk', 'night'
     */
    updateLamps(phase) {
        const intensity = {
            dawn: 0.3,
            day: 0.0,
            dusk: 0.6,
            night: 1.0,
        }[phase] || 0.0;

        for (const light of this.lampLights) {
            light.intensity = intensity;
        }
    }

    // ---------- Door Markers ----------

    _buildDoorMarkers(buildings) {
        // Door openings are created by the gap in the south wall
        // (see _createBuilding). We just add a small step/threshold
        // outside the door for visual cue.
        for (const b of buildings) {
            if (b.door_x == null || b.door_z == null) continue;

            const doorX = b.door_x + 0.5;
            const southEdgeZ = b.z + b.height;

            const stepGeo = new THREE.BoxGeometry(0.7, 0.06, 0.2);
            const stepMat = new THREE.MeshStandardMaterial({
                color: 0x888888,
                roughness: 0.9,
            });
            const step = new THREE.Mesh(stepGeo, stepMat);
            step.position.set(doorX, 0.03, southEdgeZ);
            step.receiveShadow = true;
            this.buildingGroup.add(step);
        }
    }

    // ---------- Day/Night Lighting ----------

    /**
     * Update scene lighting based on time data from server.
     * @param {THREE.DirectionalLight} sunLight
     * @param {THREE.AmbientLight} ambientLight
     * @param {THREE.Scene} scene
     * @param {Object} timeData — { phase, sun_angle, day_progress }
     */
    static updateLighting(sunLight, ambientLight, scene, timeData) {
        if (!timeData) return;

        const phase = timeData.phase;
        const sunAngle = timeData.sun_angle || 0;

        // Sun position follows an arc
        const radius = 40;
        sunLight.position.x = Math.cos(sunAngle) * radius;
        sunLight.position.y = Math.sin(sunAngle) * radius;
        sunLight.position.z = 10;

        // Phase-based colours and intensities
        switch (phase) {
            case 'dawn':
                sunLight.color.setHex(0xffaa66);
                sunLight.intensity = 0.6;
                ambientLight.color.setHex(0x404070);
                ambientLight.intensity = 0.4;
                scene.background.setHex(0xffa07a);
                scene.fog.color.setHex(0xffa07a);
                break;
            case 'day':
                sunLight.color.setHex(0xffeedd);
                sunLight.intensity = 1.0;
                ambientLight.color.setHex(0x404060);
                ambientLight.intensity = 0.6;
                scene.background.setHex(0x87ceeb);
                scene.fog.color.setHex(0x87ceeb);
                break;
            case 'dusk':
                sunLight.color.setHex(0xff6633);
                sunLight.intensity = 0.5;
                ambientLight.color.setHex(0x303050);
                ambientLight.intensity = 0.35;
                scene.background.setHex(0xcc6644);
                scene.fog.color.setHex(0xcc6644);
                break;
            case 'night':
                sunLight.color.setHex(0x223355);
                sunLight.intensity = 0.15;
                ambientLight.color.setHex(0x101030);
                ambientLight.intensity = 0.2;
                scene.background.setHex(0x0a0a2a);
                scene.fog.color.setHex(0x0a0a2a);
                break;
        }
    }

    // ---------- Helpers ----------

    _clearGroup(group) {
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            // Handle groups within groups
            if (child.children) {
                child.traverse(c => {
                    if (c.geometry) c.geometry.dispose();
                    if (c.material) c.material.dispose();
                });
            }
        }
    }
}
