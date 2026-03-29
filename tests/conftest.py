"""
Shared test fixtures for Smallville 2.

Provides mock LLM responses, sample NPC data, and test world configurations.
"""

import pytest


@pytest.fixture
def sample_npc_data():
    """Minimal NPC data for unit tests."""
    return {
        "npc_id": "blacksmith_1",
        "name": "Thorin",
        "age": 45,
        "personality_traits": ["gruff", "honest", "hardworking"],
        "backstory": "A veteran blacksmith who settled in Smallville after years of travelling.",
        "occupation": "blacksmith",
        "location": "smallville:market:blacksmith_shop:anvil",
        "home": "smallville:residential:thorin_house:bed",
        "health": 1.0,
        "energy": 0.8,
        "hunger": 0.3,
        "long_term_goals": ["Master the art of enchanted weapons", "Train an apprentice"],
        "short_term_goals": ["Finish the merchant's sword order"],
        "gold": 150,
        "inventory": {"iron_ingot": 10, "coal": 5, "steel_sword": 1},
        "skills": {"smithing": 0.9, "trading": 0.4, "combat": 0.5},
    }


@pytest.fixture
def sample_world_config():
    """Minimal world configuration for unit tests."""
    return {
        "population": 5,
        "terrain": "riverside",
        "has_ruler": False,
        "economy": "mixed",
        "seed": 42,
    }


@pytest.fixture
def mock_llm_response():
    """Factory fixture for mock LLM responses."""
    responses = {
        "daily_plan": (
            "1. Wake up at 6:00 AM\n"
            "2. Open the forge and work on sword orders until noon\n"
            "3. Have lunch at the tavern\n"
            "4. Continue smithing until 5:00 PM\n"
            "5. Visit the market to buy supplies\n"
            "6. Return home and rest"
        ),
        "importance_score": "7",
        "reflection": (
            "I've been working too hard lately and haven't spent enough "
            "time getting to know the new merchant in town. Perhaps I should "
            "visit their shop tomorrow."
        ),
    }
    return responses


@pytest.fixture
def sample_event_rules():
    """Sample event rules for testing the event impact system."""
    return [
        {
            "event_type": "trade_completed",
            "conditions": [],
            "effects": [
                {"type": "modify_sentiment", "dimension": "trust", "delta": 5},
            ],
            "scope": "individual",
        },
        {
            "event_type": "war_declared",
            "conditions": [],
            "effects": [
                {"type": "modify_global", "param": "aggression_modifier", "delta": 30},
            ],
            "scope": "world",
        },
    ]
