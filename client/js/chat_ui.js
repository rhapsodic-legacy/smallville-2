/**
 * Chat UI — text input for talking to nearby NPCs.
 *
 * Press E near an NPC to open chat. Type message, press Enter to send.
 * Conversation history scrolls. Press Escape or E again to close.
 */

const MAX_HISTORY = 50;

export class ChatUI {
    /**
     * @param {Function} sendMessage — WebSocket send function
     * @param {Function} onOpenChange — callback(isOpen) when chat opens/closes
     */
    constructor(sendMessage, onOpenChange) {
        this.sendMessage = sendMessage;
        this.onOpenChange = onOpenChange || (() => {});

        this.panel = document.getElementById('chat-panel');
        this.history = document.getElementById('chat-history');
        this.input = document.getElementById('chat-input');

        this.isOpen = false;
        this.targetNpcId = null;
        this.targetNpcName = null;
        this._nearbyNpcs = [];

        // Bind events
        this.input.addEventListener('keydown', (e) => this._onInputKey(e));

        // E key to open/close (handled by main.js, but we expose toggle)
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.code === 'KeyE') {
                e.preventDefault();
                this.toggle();
            }
        });
    }

    /** Update list of nearby NPCs (from player controls). */
    setNearbyNpcs(npcs) {
        this._nearbyNpcs = npcs || [];
    }

    /** Open chat with the closest NPC, or close if already open. */
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this._openWithClosest();
        }
    }

    /** Open chat targeting a specific NPC. */
    openWith(npcId, npcName) {
        this.targetNpcId = npcId;
        this.targetNpcName = npcName;
        this.isOpen = true;
        this.panel.classList.remove('hidden');
        this.input.placeholder = `Say something to ${npcName}...`;
        this.input.focus();
        this.onOpenChange(true);

        this._appendSystem(`Talking to ${npcName}. Type a message and press Enter.`);
    }

    /** Close the chat panel. */
    close() {
        const wasTargeting = this.targetNpcId;
        this.isOpen = false;
        this.targetNpcId = null;
        this.targetNpcName = null;
        this.panel.classList.add('hidden');
        this.input.value = '';
        this.input.blur();
        this.onOpenChange(false);
        // Tell the server to release the NPC from the conversation so
        // they resume their schedule instead of standing there waiting.
        if (wasTargeting) {
            this.sendMessage({ type: 'player_chat_close', npc_id: wasTargeting });
        }
    }

    /** Handle incoming chat response from server. */
    handleResponse(data) {
        // Drop responses that belong to a different NPC than the one
        // currently open. This prevents a lagging background LLM reply
        // from the previous NPC landing in the new NPC's chat window.
        if (data.npc_id && this.targetNpcId && data.npc_id !== this.targetNpcId) {
            return;
        }
        // Also drop if the chat panel is closed entirely.
        if (!this.isOpen) {
            return;
        }

        // Remove thinking indicator when response arrives
        this._removeThinking();

        if (data.npc_name && data.message) {
            this._appendNpc(data.npc_name, data.message);
        }
        // `note` carries a soft contextual line (e.g. "Bran had
        // already stepped away") delivered with a reply that arrived
        // after the NPC walked out of range. Show it beneath the
        // message so the player understands why the chat is closing.
        if (data.note) {
            this._appendSystem(data.note);
        }
        if (data.error) {
            this._appendSystem(data.error);
        }
        if (data.ended) {
            this._appendSystem('The conversation has ended.');
            // Auto-close the panel so the player can immediately start a
            // new chat with someone else. Staying open on an ended
            // conversation is confusing (looks like an active chat).
            setTimeout(() => this.close(), 800);
        }
    }

    /** Handle conversation broadcasts from tick (NPC-NPC or player-NPC). */
    handleConversationUpdate(conversations) {
        // Show any conversation that involves the player
        if (!conversations) return;
        for (const conv of conversations) {
            if (conv.participants && conv.participants.includes('player')) {
                for (const ex of (conv.new_exchanges || [])) {
                    if (ex.speaker_id !== 'player') {
                        this._appendNpc(ex.speaker_name, ex.message);
                    }
                }
            }
        }
    }

    // --- Private ---

    _openWithClosest() {
        if (this._nearbyNpcs.length === 0) {
            // Brief flash message
            this._showToast('No one nearby to talk to');
            return;
        }
        // Pick the closest NPC
        const closest = this._nearbyNpcs[0];
        this.openWith(closest.npc_id, closest.name);
    }

    _onInputKey(event) {
        if (event.code === 'Escape') {
            event.preventDefault();
            this.close();
            return;
        }
        // Close on E only when input is empty (avoids closing while typing)
        if (event.code === 'KeyE' && !this.input.value) {
            event.preventDefault();
            this.close();
            return;
        }
        if (event.code === 'Enter' && this.input.value.trim()) {
            event.preventDefault();
            const message = this.input.value.trim();
            this.input.value = '';

            // Display player message
            this._appendPlayer(message);

            // Show thinking indicator
            this._showThinking();

            // Send to server
            this.sendMessage({
                type: 'player_chat',
                npc_id: this.targetNpcId,
                message: message,
            });
        }
        // Stop propagation so WASD doesn't move during typing
        event.stopPropagation();
    }

    _appendPlayer(text) {
        const div = document.createElement('div');
        div.className = 'chat-message chat-player';
        div.innerHTML = `<strong>You:</strong> ${this._escapeHtml(text)}`;
        this._append(div);
    }

    _appendNpc(name, text) {
        const div = document.createElement('div');
        div.className = 'chat-message chat-npc';
        div.innerHTML = `<strong>${this._escapeHtml(name)}:</strong> ${this._escapeHtml(text)}`;
        this._append(div);
    }

    _appendSystem(text) {
        const div = document.createElement('div');
        div.className = 'chat-message chat-system';
        div.textContent = text;
        this._append(div);
    }

    _append(element) {
        this.history.appendChild(element);
        // Trim old messages
        while (this.history.children.length > MAX_HISTORY) {
            this.history.removeChild(this.history.firstChild);
        }
        this.history.scrollTop = this.history.scrollHeight;
    }

    _showThinking() {
        this._removeThinking();
        const div = document.createElement('div');
        div.className = 'chat-message chat-thinking';
        div.id = 'chat-thinking-indicator';
        div.innerHTML = `<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>`;
        this._append(div);
    }

    _removeThinking() {
        const el = this.history.querySelector('#chat-thinking-indicator');
        if (el) el.remove();
    }

    _showToast(text) {
        const toast = document.createElement('div');
        toast.className = 'chat-toast';
        toast.textContent = text;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
