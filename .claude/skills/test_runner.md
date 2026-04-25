---
name: test_runner
description: Run automated test suites. Claude runs these automatically — never ask the user to run tests manually.
---

# Skill: Test Runner

## Critical Rule
**Claude runs all tests automatically.** Never ask the user to run tests, observe behaviour, or manually verify. Run the relevant pipeline, fix failures, re-run until green, then report results.

## Test Pipelines

### Movement & Pathfinding (auto-run after spatial/movement changes)

    python3 tests/simulation/test_npc_movement.py

- 11 automated checks: doors, paths, building clearance, water/wall avoidance, stuck detection, speed variety, overlap resolution
- Run after ANY change to: pathfinding, generator, spatial_awareness, execute, models, npc_renderer
- If failures: fix and re-run before telling the user anything

### Unit Tests (fast, no LLM)

    pytest tests/unit/ -v

- Mock all LLM calls with fixtures
- Test game logic, data models, pathfinding, memory CRUD
- Should complete in <30 seconds

### Integration Tests (moderate, uses Haiku)

    pytest tests/integration/ -v

- Hit real Claude Haiku API for cognition pipeline validation
- Test full perceive → retrieve → plan → reflect → execute cycle
- Budget: ~$0.10 per full run

### Simulation Tests (slow, headless)

    pytest tests/simulation/ -v --timeout=600

- Run world simulation without human player
- Validate NPC behaviours over multiple game days

## Workflow
1. Make code changes
2. Run relevant test pipeline(s) automatically
3. If failures → fix → re-run
4. Only after ALL CLEAR → restart server (if needed) and report to user

## Adding New Pipelines
When building any new system, create a corresponding test file in `tests/` and register it here. Every system should have an automated validation pipeline.

## Test Conventions
- Test files mirror source structure: core/npc/models.py → tests/unit/test_npc_models.py
- Fixtures in tests/conftest.py for shared test data
- Use @pytest.mark.integration for tests that need real API
- Use @pytest.mark.simulation for long-running tests
