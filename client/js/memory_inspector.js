/**
 * Memory Inspector — debug panel for observing NPC memory system.
 *
 * Shows: system stats, per-NPC memory browser, live activity feed.
 * Toggle with 'M' key. Fetches data via REST endpoints.
 */

const POLL_INTERVAL = 3000; // ms between stat refreshes
const ACTIVITY_POLL = 5000; // ms between activity feed refreshes

export class MemoryInspector {
    constructor(sendMessage) {
        this._sendMessage = sendMessage;
        this._visible = false;
        this._pollTimer = null;
        this._activityTimer = null;
        this._selectedNpcId = null;

        this._buildDOM();
        this._bindEvents();
    }

    // ---------- DOM construction ----------

    _buildDOM() {
        this.panel = document.createElement('div');
        this.panel.id = 'memory-inspector';
        this.panel.className = 'memory-inspector hidden';
        this.panel.innerHTML = `
            <div class="mi-header">
                <h3>Memory Inspector</h3>
                <span class="mi-close">&times;</span>
            </div>
            <div class="mi-tabs">
                <button class="mi-tab active" data-tab="overview">Overview</button>
                <button class="mi-tab" data-tab="npc">NPC Browser</button>
                <button class="mi-tab" data-tab="activity">Activity Feed</button>
            </div>
            <div class="mi-content">
                <div class="mi-tab-content active" data-tab="overview">
                    <div class="mi-stats">Loading...</div>
                </div>
                <div class="mi-tab-content" data-tab="npc">
                    <div class="mi-npc-list"></div>
                    <div class="mi-npc-detail"></div>
                </div>
                <div class="mi-tab-content" data-tab="activity">
                    <div class="mi-activity-feed"></div>
                </div>
            </div>
        `;
        document.getElementById('game-container').appendChild(this.panel);
    }

    _bindEvents() {
        // Close button
        this.panel.querySelector('.mi-close').addEventListener('click', () => {
            this.toggle();
        });

        // Tab switching
        this.panel.querySelectorAll('.mi-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this._switchTab(tabName);
            });
        });
    }

    // ---------- Toggle ----------

    toggle() {
        this._visible = !this._visible;
        this.panel.classList.toggle('hidden', !this._visible);

        if (this._visible) {
            this._startPolling();
            this._fetchStats();
            this._fetchNpcList();
            this._fetchActivity();
        } else {
            this._stopPolling();
        }
    }

    get visible() {
        return this._visible;
    }

    // ---------- Tabs ----------

    _switchTab(tabName) {
        this.panel.querySelectorAll('.mi-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabName);
        });
        this.panel.querySelectorAll('.mi-tab-content').forEach(c => {
            c.classList.toggle('active', c.dataset.tab === tabName);
        });

        if (tabName === 'npc') this._fetchNpcList();
        if (tabName === 'activity') this._fetchActivity();
    }

    // ---------- Polling ----------

    _startPolling() {
        this._pollTimer = setInterval(() => this._fetchStats(), POLL_INTERVAL);
        this._activityTimer = setInterval(() => this._fetchActivity(), ACTIVITY_POLL);
    }

    _stopPolling() {
        if (this._pollTimer) clearInterval(this._pollTimer);
        if (this._activityTimer) clearInterval(this._activityTimer);
        this._pollTimer = null;
        this._activityTimer = null;
    }

    // ---------- Data fetching ----------

    async _fetchStats() {
        try {
            const resp = await fetch('/api/memory/stats');
            const data = await resp.json();
            this._renderStats(data);
        } catch (e) {
            console.warn('Memory inspector: stats fetch failed', e);
        }
    }

    async _fetchNpcList() {
        try {
            const resp = await fetch('/api/memory/npcs');
            const data = await resp.json();
            this._renderNpcList(data);
        } catch (e) {
            console.warn('Memory inspector: NPC list fetch failed', e);
        }
    }

    async _fetchNpcMemory(npcId) {
        try {
            const resp = await fetch(`/api/memory/npc/${npcId}`);
            const data = await resp.json();
            this._selectedNpcId = npcId;
            this._renderNpcDetail(data);
        } catch (e) {
            console.warn('Memory inspector: NPC memory fetch failed', e);
        }
    }

    async _fetchActivity() {
        try {
            const resp = await fetch('/api/memory/stats');
            const data = await resp.json();
            this._renderActivity(data.activity || []);
        } catch (e) {
            console.warn('Memory inspector: activity fetch failed', e);
        }
    }

    // ---------- Rendering ----------

    _renderStats(data) {
        const stats = data.stats || {};
        const structured = stats.structured || {};
        const episodic = stats.episodic || {};
        const spatial = stats.spatial || {};

        const el = this.panel.querySelector('.mi-stats');
        el.innerHTML = `
            <div class="mi-stat-grid">
                <div class="mi-stat-card">
                    <div class="mi-stat-label">Structured Memory</div>
                    <div class="mi-stat-value">${structured.total_facts || 0}</div>
                    <div class="mi-stat-sub">facts stored</div>
                </div>
                <div class="mi-stat-card">
                    <div class="mi-stat-label">Episodic Memory</div>
                    <div class="mi-stat-value">${episodic.total_memories || 0}</div>
                    <div class="mi-stat-sub">${episodic.backend || 'unknown'}</div>
                </div>
                <div class="mi-stat-card">
                    <div class="mi-stat-label">Spatial Memory</div>
                    <div class="mi-stat-value">${spatial.npcs_with_spatial || 0}</div>
                    <div class="mi-stat-sub">NPCs with spatial knowledge</div>
                </div>
                <div class="mi-stat-card">
                    <div class="mi-stat-label">Active Goals</div>
                    <div class="mi-stat-value">${structured.active_goals || 0}</div>
                    <div class="mi-stat-sub">across all NPCs</div>
                </div>
                <div class="mi-stat-card">
                    <div class="mi-stat-label">Events Recorded</div>
                    <div class="mi-stat-value">${structured.total_events || 0}</div>
                    <div class="mi-stat-sub">conversations, observations</div>
                </div>
                <div class="mi-stat-card">
                    <div class="mi-stat-label">Sectors Known</div>
                    <div class="mi-stat-value">${spatial.total_sectors_known || 0}</div>
                    <div class="mi-stat-sub">${spatial.total_arenas_known || 0} arenas</div>
                </div>
            </div>
        `;
    }

    _renderNpcList(npcs) {
        const el = this.panel.querySelector('.mi-npc-list');
        if (!npcs || npcs.length === 0) {
            el.innerHTML = '<div class="mi-empty">No NPCs found</div>';
            return;
        }

        el.innerHTML = npcs.map(npc => `
            <div class="mi-npc-row ${npc.npc_id === this._selectedNpcId ? 'selected' : ''}"
                 data-npc-id="${npc.npc_id}">
                <span class="mi-npc-name">${npc.name}</span>
                <span class="mi-npc-occ">${npc.occupation}</span>
                <span class="mi-npc-tier">T${npc.cognition_tier}</span>
                <span class="mi-npc-mem">${npc.episodic_count} mem</span>
            </div>
        `).join('');

        // Bind click handlers
        el.querySelectorAll('.mi-npc-row').forEach(row => {
            row.addEventListener('click', () => {
                this._fetchNpcMemory(row.dataset.npcId);
            });
        });
    }

    _renderNpcDetail(data) {
        const el = this.panel.querySelector('.mi-npc-detail');
        if (!data || data.error) {
            el.innerHTML = `<div class="mi-empty">${data?.error || 'No data'}</div>`;
            return;
        }

        const facts = data.facts || [];
        const memories = data.recent_memories || [];
        const goals = data.goals || [];
        const spatial = data.spatial || {};

        el.innerHTML = `
            <div class="mi-detail-section">
                <h4>Episodic Memories (${data.episodic_count || 0} total)</h4>
                <div class="mi-memory-list">
                    ${memories.length === 0 ? '<div class="mi-empty">No memories yet</div>' :
                        memories.map(m => `
                            <div class="mi-memory-item mi-cat-${m.category}">
                                <div class="mi-mem-header">
                                    <span class="mi-mem-cat">${m.category}</span>
                                    <span class="mi-mem-imp" title="importance">
                                        ${'●'.repeat(Math.round(m.importance * 5))}${'○'.repeat(5 - Math.round(m.importance * 5))}
                                    </span>
                                    <span class="mi-mem-time">${this._formatGameTime(m.game_time)}</span>
                                </div>
                                <div class="mi-mem-desc">${this._escapeHtml(m.description)}</div>
                            </div>
                        `).join('')
                    }
                </div>
            </div>

            <div class="mi-detail-section">
                <h4>Known Facts (${facts.length})</h4>
                <div class="mi-fact-list">
                    ${facts.length === 0 ? '<div class="mi-empty">No facts known</div>' :
                        facts.map(f => `
                            <div class="mi-fact-item">
                                ${this._escapeHtml(f.subject)} <em>${this._escapeHtml(f.predicate)}</em> ${this._escapeHtml(f.object)}
                            </div>
                        `).join('')
                    }
                </div>
            </div>

            <div class="mi-detail-section">
                <h4>Goals (${goals.length})</h4>
                <div class="mi-goal-list">
                    ${goals.length === 0 ? '<div class="mi-empty">No active goals</div>' :
                        goals.map(g => `
                            <div class="mi-goal-item">
                                <span class="mi-goal-status">${g.status}</span>
                                ${this._escapeHtml(g.description)}
                            </div>
                        `).join('')
                    }
                </div>
            </div>

            <div class="mi-detail-section">
                <h4>Spatial Knowledge</h4>
                <div class="mi-spatial-tree">
                    ${Object.keys(spatial).length === 0 ? '<div class="mi-empty">No spatial knowledge</div>' :
                        Object.entries(spatial).map(([sector, sk]) => `
                            <div class="mi-spatial-sector">
                                <strong>${sector}</strong>
                                ${Object.entries(sk.arenas || {}).map(([arena, ak]) => `
                                    <div class="mi-spatial-arena">
                                        ${arena}
                                        ${ak.objects?.length ? ': ' + ak.objects.join(', ') : ''}
                                    </div>
                                `).join('')}
                            </div>
                        `).join('')
                    }
                </div>
            </div>
        `;
    }

    _renderActivity(activity) {
        const el = this.panel.querySelector('.mi-activity-feed');
        if (!activity || activity.length === 0) {
            el.innerHTML = '<div class="mi-empty">No activity yet — memories will appear as NPCs perceive and converse.</div>';
            return;
        }

        el.innerHTML = activity.map(event => `
            <div class="mi-activity-item mi-event-${event.event_type}">
                <div class="mi-activity-header">
                    <span class="mi-event-type">${event.event_type}</span>
                    <span class="mi-event-time">t=${Math.round(event.game_time)}</span>
                </div>
                <div class="mi-activity-desc">${this._escapeHtml(event.description?.substring(0, 200))}</div>
                ${event.participants?.length ? `
                    <div class="mi-activity-participants">
                        ${event.participants.map(p => `<span class="mi-participant">${p}</span>`).join(' ')}
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    _formatGameTime(gameMinutes) {
        if (gameMinutes === undefined || gameMinutes === null) return '';
        const totalMinutes = Math.round(gameMinutes);
        const day = Math.floor(totalMinutes / 1440) + 1; // 1440 = 24*60
        const minuteOfDay = totalMinutes % 1440;
        const hour = Math.floor(minuteOfDay / 60);
        const minute = minuteOfDay % 60;
        const hh = String(hour).padStart(2, '0');
        const mm = String(minute).padStart(2, '0');
        return `Day ${day} ${hh}:${mm}`;
    }
}
