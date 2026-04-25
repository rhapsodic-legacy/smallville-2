"""
Pytest plumbing for the narrative sim-test suite.

Re-exports the `sim` fixture from framework.py so every test file
under this directory can accept `sim: NarrativeSim` as a parameter
without having to import the fixture themselves.
"""

from tests.simulation.narrative.framework import sim  # noqa: F401
