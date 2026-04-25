# Skill: Memory Operations

## When to Use
When working on the memory system — storing, retrieving, or reflecting on NPC memories.

## Hybrid Memory Architecture

### Structured Memory (SQLite)
For hard facts and queryable relationships:
- "Player owes me 50 gold"
- "I am allied with the blacksmith"
- "The church is 60% built"
- "War has been declared"

Schema tables: facts, relationships, goals, world_state

### Episodic Memory (ChromaDB)
For experiential/subjective memories:
- "The player seemed dishonest last time"
- "The harvest festival was joyful"
- "Bob proposed to Martha at sunset"

Stored as text with embeddings, scored by recency + importance + relevance.

## Memory Formation Pipeline
1. **Observe** — Perception module detects event on nearby tiles
2. **Score importance** — LLM rates 1 to 10 (poignancy)
3. **Classify** — Is this a hard fact (structured) or experience (episodic)?
4. **Store** — Write to appropriate store with timestamp and metadata
5. **Index** — Keywords for structured, embeddings for episodic

## Retrieval (Upgraded from Stanford)
Score = (recency_weight * recency) + (relevance_weight * relevance) + (importance_weight * importance)

- **Recency:** Exponential decay from last access (0.99^age)
- **Relevance:** Cosine similarity of embedding to query
- **Importance:** Normalised poignancy score

Return top N memories combining both stores.

## Reflection Trigger
When accumulated importance of new observations exceeds threshold (default 150):
1. Generate 3 focal points from recent memories with high importance
2. Retrieve 30 memories per focal point
3. LLM synthesises 5 insights at a higher level per focal point
4. Store insights as new thought nodes (become retrievable memories)

## Key Rules
- Never store duplicate memories — check before inserting
- Update last_accessed on every retrieval
- Memories expire after configurable duration (default 30 game days)
- After conversations: always generate planning thought + memo thought
