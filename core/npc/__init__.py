"""NPC module — data models, tiered cognition, LLM integration."""

from core.npc.models import NPC, PersonalityTraits, ActivityState, Direction, ScheduleEntry
from core.npc.manager import NPCManager
from core.npc.llm_client import ClaudeProvider, MockProvider, LLMProvider

__all__ = [
    "NPC",
    "PersonalityTraits",
    "ActivityState",
    "Direction",
    "ScheduleEntry",
    "NPCManager",
    "ClaudeProvider",
    "MockProvider",
    "LLMProvider",
]
