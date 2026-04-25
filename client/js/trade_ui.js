/**
 * Trading UI — player inventory display and trade proposal interface.
 *
 * Press T near a merchant/NPC to open trade panel.
 * Select items to offer/request, adjust gold, and propose.
 * NPC evaluates via the existing trade system.
 */

export class TradeUI {
    /**
     * @param {Function} sendMessage — WebSocket send function
     * @param {Function} onOpenChange — callback(isOpen)
     */
    constructor(sendMessage, onOpenChange) {
        this.sendMessage = sendMessage;
        this.onOpenChange = onOpenChange || (() => {});

        this.isOpen = false;
        this.targetNpcId = null;
        this.targetNpcName = null;

        // Player state (updated from server)
        this.playerGold = 0;
        this.playerInventory = {};

        // Trade form state
        this.offerItems = {};
        this.offerGold = 0;
        this.requestItems = {};
        this.requestGold = 0;

        this._nearbyNpcs = [];

        // Create DOM
        this._panel = this._createPanel();
        document.getElementById('game-container').appendChild(this._panel);

        // T key to open/close
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.code === 'KeyT') {
                e.preventDefault();
                this.toggle();
            }
        });
    }

    setNearbyNpcs(npcs) {
        this._nearbyNpcs = npcs || [];
    }

    updatePlayerState(data) {
        if (data.gold !== undefined) this.playerGold = data.gold;
        if (data.inventory) this.playerInventory = data.inventory;
        if (this.isOpen) this._renderInventory();
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this._openWithClosest();
        }
    }

    openWith(npcId, npcName) {
        this.targetNpcId = npcId;
        this.targetNpcName = npcName;
        this.offerItems = {};
        this.offerGold = 0;
        this.requestItems = {};
        this.requestGold = 0;
        this.isOpen = true;
        this._panel.classList.remove('hidden');
        this._render();
        this.onOpenChange(true);
    }

    close() {
        this.isOpen = false;
        this.targetNpcId = null;
        this._panel.classList.add('hidden');
        this.onOpenChange(false);
    }

    /** Handle trade response from server. */
    handleResponse(data) {
        const statusEl = this._panel.querySelector('.trade-status');
        if (!statusEl) return;

        if (data.accepted) {
            statusEl.textContent = `${data.npc_name || 'NPC'} accepted the trade!`;
            statusEl.className = 'trade-status trade-accepted';
            if (data.gold !== undefined) this.playerGold = data.gold;
            if (data.inventory) this.playerInventory = data.inventory;
            this._renderInventory();
        } else {
            statusEl.textContent = data.reason || 'Trade rejected.';
            statusEl.className = 'trade-status trade-rejected';
        }
    }

    // --- Private ---

    _openWithClosest() {
        if (this._nearbyNpcs.length === 0) return;
        const closest = this._nearbyNpcs[0];
        this.openWith(closest.npc_id, closest.name);
    }

    _render() {
        const content = this._panel.querySelector('.trade-content');
        content.innerHTML = '';

        // Header
        const header = document.createElement('div');
        header.className = 'trade-header';
        header.innerHTML = `
            <h3>Trade with ${this._esc(this.targetNpcName)}</h3>
            <span class="trade-close" title="Close">&times;</span>
        `;
        header.querySelector('.trade-close').addEventListener('click', () => this.close());
        content.appendChild(header);

        // Inventory section
        const inv = document.createElement('div');
        inv.className = 'trade-inventory';
        inv.innerHTML = `<h4>Your inventory — Gold: ${this.playerGold}</h4>`;
        const invList = document.createElement('div');
        invList.className = 'trade-inv-list';
        content.appendChild(inv);
        inv.appendChild(invList);
        this._renderInventory();

        // Offer section
        const offerSection = this._createSection('You offer:', 'offer');
        content.appendChild(offerSection);

        // Request section
        const requestSection = this._createSection('You request:', 'request');
        content.appendChild(requestSection);

        // Gold inputs
        const goldRow = document.createElement('div');
        goldRow.className = 'trade-gold-row';
        goldRow.innerHTML = `
            <label>Offer gold: <input type="number" min="0" max="${this.playerGold}" value="0" class="trade-offer-gold" /></label>
            <label>Request gold: <input type="number" min="0" value="0" class="trade-request-gold" /></label>
        `;
        content.appendChild(goldRow);

        goldRow.querySelector('.trade-offer-gold').addEventListener('change', (e) => {
            this.offerGold = Math.max(0, Math.min(this.playerGold, parseInt(e.target.value) || 0));
        });
        goldRow.querySelector('.trade-request-gold').addEventListener('change', (e) => {
            this.requestGold = Math.max(0, parseInt(e.target.value) || 0);
        });

        // Propose button
        const btn = document.createElement('button');
        btn.className = 'trade-propose-btn';
        btn.textContent = 'Propose Trade';
        btn.addEventListener('click', () => this._propose());
        content.appendChild(btn);

        // Status
        const status = document.createElement('div');
        status.className = 'trade-status';
        content.appendChild(status);
    }

    _renderInventory() {
        const invList = this._panel.querySelector('.trade-inv-list');
        if (!invList) return;
        invList.innerHTML = '';
        const entries = Object.entries(this.playerInventory);
        if (entries.length === 0) {
            invList.innerHTML = '<span class="trade-empty">No items</span>';
            return;
        }
        for (const [item, qty] of entries) {
            const el = document.createElement('span');
            el.className = 'trade-inv-item';
            el.textContent = `${item}: ${qty}`;
            invList.appendChild(el);
        }
    }

    _createSection(label, prefix) {
        const section = document.createElement('div');
        section.className = `trade-section trade-${prefix}`;
        section.innerHTML = `<h4>${label}</h4>`;
        const itemInput = document.createElement('div');
        itemInput.className = 'trade-item-input';
        itemInput.innerHTML = `
            <input type="text" placeholder="Item name" class="trade-${prefix}-item" />
            <input type="number" min="1" value="1" class="trade-${prefix}-qty" />
            <button class="trade-${prefix}-add">Add</button>
        `;
        section.appendChild(itemInput);

        const list = document.createElement('div');
        list.className = `trade-${prefix}-list`;
        section.appendChild(list);

        itemInput.querySelector(`.trade-${prefix}-add`).addEventListener('click', () => {
            const itemEl = itemInput.querySelector(`.trade-${prefix}-item`);
            const qtyEl = itemInput.querySelector(`.trade-${prefix}-qty`);
            const item = itemEl.value.trim().toLowerCase();
            const qty = parseInt(qtyEl.value) || 1;
            if (!item) return;

            const target = prefix === 'offer' ? this.offerItems : this.requestItems;
            target[item] = (target[item] || 0) + qty;
            itemEl.value = '';
            qtyEl.value = '1';
            this._renderItemList(list, target, prefix);
        });

        return section;
    }

    _renderItemList(container, items, prefix) {
        container.innerHTML = '';
        for (const [item, qty] of Object.entries(items)) {
            const el = document.createElement('span');
            el.className = 'trade-list-item';
            el.innerHTML = `${this._esc(item)} x${qty} <span class="trade-remove">&times;</span>`;
            el.querySelector('.trade-remove').addEventListener('click', () => {
                delete items[item];
                this._renderItemList(container, items, prefix);
            });
            container.appendChild(el);
        }
    }

    _propose() {
        this.sendMessage({
            type: 'player_trade',
            npc_id: this.targetNpcId,
            items_offered: { ...this.offerItems },
            gold_offered: this.offerGold,
            items_requested: { ...this.requestItems },
            gold_requested: this.requestGold,
        });
        const status = this._panel.querySelector('.trade-status');
        if (status) {
            status.textContent = 'Waiting for response...';
            status.className = 'trade-status trade-pending';
        }
    }

    _createPanel() {
        const panel = document.createElement('div');
        panel.id = 'trade-panel';
        panel.className = 'trade-panel hidden';
        const content = document.createElement('div');
        content.className = 'trade-content';
        panel.appendChild(content);
        return panel;
    }

    _esc(text) {
        const d = document.createElement('div');
        d.textContent = text || '';
        return d.innerHTML;
    }
}
