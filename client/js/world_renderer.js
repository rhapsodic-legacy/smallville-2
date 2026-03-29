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
        this.lampGroup = new THREE.Group();
        this.scene.add(this.terrainGroup);
        this.scene.add(this.buildingGroup);
        this.scene.add(this.resourceGroup);
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
        this._buildLampPosts(worldData.tiles);
        if (buildings) this._buildDoorMarkers(buildings);
    }

    clearWorld() {
        this._clearGroup(this.terrainGroup);
        this._clearGroup(this.buildingGroup);
        this._clearGroup(this.resourceGroup);
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

        const group = new THREE.Group();

        // Walls (box)
        const wallGeo = new THREE.BoxGeometry(w * 0.9, wallHeight, h * 0.9);
        const wallMat = new THREE.MeshStandardMaterial({
            color: wallColour,
            roughness: 0.8,
        });
        const walls = new THREE.Mesh(wallGeo, wallMat);
        walls.position.y = wallHeight / 2;
        walls.castShadow = true;
        walls.receiveShadow = true;
        group.add(walls);

        // Roof (pyramid / cone)
        if (bType === 'church') {
            // Steeple
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
            // Flat canopy
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
            // Pitched roof (use a box rotated to approximate)
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
        const doorGeo = new THREE.BoxGeometry(0.5, 1.2, 0.1);
        const doorMat = new THREE.MeshStandardMaterial({
            color: 0x3a2010,
            roughness: 0.7,
        });

        for (const b of buildings) {
            if (b.door_x == null || b.door_z == null) continue;

            // Door tile is on the building's south wall (last row of footprint).
            // Position the door mesh flush with the building's south face.
            const doorX = b.door_x + 0.5;
            // Building south wall face: mesh centre z + half wall depth
            const southFaceZ = b.z + b.height / 2 + (b.height * 0.9) / 2;
            const doorZ = southFaceZ;

            // Direction from building centre to door (outward)
            const cx = b.x + b.width / 2;
            const cz = b.z + b.height / 2;
            const faceDx = doorX - cx;
            const faceDz = doorZ - cz;

            const door = new THREE.Mesh(doorGeo, doorMat);
            door.position.set(doorX, 0.6, doorZ);

            // Face the door outward (away from building centre)
            door.lookAt(
                doorX + faceDx,
                0.6,
                doorZ + faceDz,
            );

            door.castShadow = true;
            this.buildingGroup.add(door);

            // Small step/threshold in front of the door
            const stepGeo = new THREE.BoxGeometry(0.6, 0.08, 0.3);
            const stepMat = new THREE.MeshStandardMaterial({
                color: 0x888888,
                roughness: 0.9,
            });
            const step = new THREE.Mesh(stepGeo, stepMat);
            // Step is slightly in front of door (toward the outside)
            const stepOffset = 0.2;
            const faceLen = Math.sqrt(faceDx * faceDx + faceDz * faceDz) || 1;
            step.position.set(
                doorX + (faceDx / faceLen) * stepOffset,
                0.04,
                doorZ + (faceDz / faceLen) * stepOffset,
            );
            step.rotation.y = door.rotation.y;
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
