/**
 * HUD — time display, player stats, minimap, and notification feed.
 *
 * Replaces the simple inline HUD from main.js with a proper module.
 */

const MINIMAP_SIZE = 150;
const MINIMAP_PADDING = 12;
const NOTIFICATION_DURATION = 5000;
const MAX_NOTIFICATIONS = 5;

// Phase E — category → colour map for sparkles and the HUD dots.
// Red for accusations, gold for commitments, green for shared town
// events, blue for conversation turns, purple for relayed claims,
// silver for generic observations.
export const MEMORY_CATEGORY_COLOURS = {
    accusation:        { css: '#ff5a5a', hex: 0xff5a5a },
    commitment:        { css: '#ffd23f', hex: 0xffd23f },
    relayed_claim:     { css: '#c77dff', hex: 0xc77dff },
    town_event:        { css: '#7cf29c', hex: 0x7cf29c },
    town_failure:      { css: '#ff9f6b', hex: 0xff9f6b },
    town_agenda:       { css: '#7cc4ff', hex: 0x7cc4ff },
    conversation:      { css: '#b7c7d9', hex: 0xb7c7d9 },
    conversation_turn: { css: '#b7c7d9', hex: 0xb7c7d9 },
    reflection:        { css: '#ffeb99', hex: 0xffeb99 },
    observation:       { css: '#cfd8dc', hex: 0xcfd8dc },
};

const DEFAULT_MEMORY_COLOUR = { css: '#cfd8dc', hex: 0xcfd8dc };

export function memoryCategoryColour(category) {
    return MEMORY_CATEGORY_COLOURS[category] || DEFAULT_MEMORY_COLOUR;
}

// Importance threshold above which a memory_formed event also
// produces a notification feed entry. Matches the server's
// MEMORY_EVENT_THRESHOLD (0.6) — anything surfacing at all is
// already notable enough to mention.
const NOTIFY_MEMORY_THRESHOLD = 0.7;

function _esc(s) {
    const d = document.createElement('div');
    d.textContent = String(s || '');
    return d.innerHTML;
}

export class HUD {
    constructor() {
        // Existing HUD elements
        this._timeEl = document.getElementById('hud-time');
        this._goldEl = document.getElementById('hud-gold');
        this._statusEl = document.getElementById('hud-status');

        // Create new HUD elements
        this._statsBar = this._createStatsBar();
        this._minimap = this._createMinimap();
        this._notifContainer = document.getElementById('notifications');
        this._nearbyList = this._createNearbyList();

        document.getElementById('hud').appendChild(this._statsBar);
        document.getElementById('hud').appendChild(this._nearbyList);
        document.getElementById('game-container').appendChild(this._minimap.canvas);

        // State
        this._worldWidth = 60;
        this._worldHeight = 60;
        this._notifications = [];

        // Phase E — per-NPC rolling cache of the most recent
        // `memory_formed` event the server broadcast. The nearby-NPC
        // list colours its bullet by this event's category and shows
        // the summary on hover.
        this._latestMemoryByNpc = new Map();
    }

    /** Record a memory_formed event from the server tick.
     *
     * Caches the latest per NPC for hover tooltip rendering and,
     * above a threshold, surfaces a notification-feed entry.
     */
    recordMemoryEvent(evt) {
        if (!evt || !evt.npc_id) return;
        this._latestMemoryByNpc.set(evt.npc_id, {
            summary: evt.summary || '',
            category: evt.category || '',
            importance: typeof evt.importance === 'number' ? evt.importance : 0,
        });
        if ((evt.importance || 0) >= NOTIFY_MEMORY_THRESHOLD) {
            const tag = evt.category ? `[${evt.category}] ` : '';
            this.notify(`${tag}${evt.summary || 'new memory'}`, 'memory');
        }
    }

    /** Update time display. */
    updateTime(timeData) {
        if (!timeData) return;
        this._timeEl.textContent = `Day ${timeData.day} — ${timeData.time} (${timeData.phase})`;
    }

    /** Update the town agenda panel.
     *
     * Hides when there are no active or proposed goals so the HUD
     * doesn't permanently reserve space. Completed goals briefly
     * linger in the "completed_recent" list so the player can see
     * what the town just achieved.
     */
    updateTownAgenda(agenda) {
        const el = document.getElementById('hud-agenda');
        if (!el) return;
        const active = (agenda && agenda.active) || [];
        const recent = (agenda && agenda.completed_recent) || [];
        if (active.length === 0 && recent.length === 0) {
            el.classList.add('hidden');
            el.innerHTML = '';
            return;
        }
        el.classList.remove('hidden');
        const parts = ['<div class="agenda-title">Town agenda</div>'];
        for (const g of active) {
            const pct = Math.min(100, Math.round(
                (g.progress / Math.max(1, g.required_contributions)) * 100));
            parts.push(
                `<div class="agenda-goal">` +
                `<div class="agenda-goal-title">${_esc(g.title)}</div>` +
                `<div class="agenda-progress">` +
                `<span class="agenda-status-${g.status}">${g.status}</span> · ` +
                `${g.progress}/${g.required_contributions} (${pct}%)` +
                `</div></div>`
            );
        }
        for (const g of recent) {
            parts.push(
                `<div class="agenda-goal agenda-status-completed">` +
                `✓ ${_esc(g.title)}</div>`
            );
        }
        el.innerHTML = parts.join('');
    }

    /** Update player stats (gold, energy, hunger). */
    updatePlayerStats(playerData) {
        if (!playerData) return;
        this._goldEl.textContent = `Gold: ${playerData.gold || 0}`;

        const energy = playerData.energy || 0;
        const hunger = playerData.hunger || 0;
        const health = playerData.health || 1;

        this._statsBar.querySelector('.hud-energy-fill').style.width = `${energy * 100}%`;
        this._statsBar.querySelector('.hud-hunger-fill').style.width = `${(1 - hunger) * 100}%`;
        this._statsBar.querySelector('.hud-health-fill').style.width = `${health * 100}%`;
    }

    /** Update minimap with NPC and player positions. */
    updateMinimap(npcs, playerData, buildings) {
        const ctx = this._minimap.ctx;
        const scale = MINIMAP_SIZE / Math.max(this._worldWidth, this._worldHeight);

        // Clear
        ctx.fillStyle = '#1a2a1a';
        ctx.fillRect(0, 0, MINIMAP_SIZE, MINIMAP_SIZE);

        // Buildings
        if (buildings) {
            ctx.fillStyle = '#555';
            for (const b of buildings) {
                ctx.fillRect(
                    b.x * scale, b.z * scale,
                    (b.width || 1) * scale, (b.height || 1) * scale
                );
            }
        }

        // NPCs
        if (npcs) {
            ctx.fillStyle = '#44aaff';
            for (const npc of npcs) {
                if (npc.is_player) continue;
                ctx.beginPath();
                ctx.arc(npc.x * scale, npc.z * scale, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Player
        if (playerData) {
            ctx.fillStyle = '#ffd700';
            ctx.beginPath();
            ctx.arc(playerData.x * scale, playerData.z * scale, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    /** Update nearby NPC list. */
    updateNearbyNpcs(nearbyNpcs) {
        this._nearbyList.innerHTML = '';
        if (!nearbyNpcs || nearbyNpcs.length === 0) {
            this._nearbyList.classList.add('hidden');
            return;
        }
        this._nearbyList.classList.remove('hidden');
        for (const npc of nearbyNpcs.slice(0, 5)) {
            const row = document.createElement('div');
            row.className = 'hud-nearby-npc';

            // Phase E.1 — if we have a cached memory_formed for this
            // NPC, render a small coloured dot keyed to the memory's
            // category and expose the summary via a hover tooltip.
            // Empty-state stays visually unchanged.
            const latest = this._latestMemoryByNpc.get(npc.npc_id);
            if (latest && latest.summary) {
                const colour = memoryCategoryColour(latest.category);
                row.classList.add('hud-nearby-npc--has-memory');
                row.style.setProperty('--memory-dot-colour', colour.css);
                row.title = (
                    `Latest memory (${latest.category || 'memory'}, `
                    + `importance ${latest.importance.toFixed(2)}):\n`
                    + latest.summary
                );
                row.innerHTML = (
                    `<span class="hud-memory-dot" `
                    + `style="background:${colour.css}"></span>`
                    + `${_esc(npc.name)} (${_esc(npc.occupation)})`
                );
            } else {
                row.textContent = `${npc.name} (${npc.occupation})`;
            }
            this._nearbyList.appendChild(row);
        }
    }

    /** Set world dimensions for minimap scaling. */
    setWorldSize(width, height) {
        this._worldWidth = width;
        this._worldHeight = height;
    }

    /** Add a notification to the feed. */
    notify(text, type = 'info') {
        const el = document.createElement('div');
        el.className = `notification notif-${type}`;
        el.textContent = text;
        this._notifContainer.appendChild(el);
        this._notifications.push(el);

        // Auto-remove
        setTimeout(() => {
            el.style.opacity = '0';
            setTimeout(() => {
                el.remove();
                this._notifications = this._notifications.filter(n => n !== el);
            }, 300);
        }, NOTIFICATION_DURATION);

        // Trim
        while (this._notifications.length > MAX_NOTIFICATIONS) {
            const old = this._notifications.shift();
            old.remove();
        }
    }

    /** Update connection status. */
    setStatus(text, color) {
        this._statusEl.textContent = text;
        this._statusEl.style.color = color;
    }

    // --- Private ---

    _createStatsBar() {
        const bar = document.createElement('div');
        bar.id = 'hud-stats';
        bar.innerHTML = `
            <div class="hud-stat-row">
                <span class="hud-stat-label">HP</span>
                <div class="hud-stat-bar"><div class="hud-health-fill hud-fill"></div></div>
            </div>
            <div class="hud-stat-row">
                <span class="hud-stat-label">EN</span>
                <div class="hud-stat-bar"><div class="hud-energy-fill hud-fill"></div></div>
            </div>
            <div class="hud-stat-row">
                <span class="hud-stat-label">FD</span>
                <div class="hud-stat-bar"><div class="hud-hunger-fill hud-fill"></div></div>
            </div>
        `;
        return bar;
    }

    _createMinimap() {
        const canvas = document.createElement('canvas');
        canvas.id = 'minimap';
        canvas.width = MINIMAP_SIZE;
        canvas.height = MINIMAP_SIZE;
        const ctx = canvas.getContext('2d');
        return { canvas, ctx };
    }

    _createNearbyList() {
        const el = document.createElement('div');
        el.id = 'hud-nearby';
        el.className = 'hidden';
        return el;
    }
}
