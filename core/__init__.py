"""
smallville_core — Self-evolving NPC ecosystem library.

Public API (target):
    from smallville_core import World, NPCManager, Overseer

    world = World.generate(population=30, terrain="riverside")
    world.tick()
    state = world.get_state()
"""

__version__ = "0.1.0"
