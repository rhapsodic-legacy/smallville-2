"""
Tests for Phase 6.3: Construction System.

Covers blueprints, build phases, construction sites, resource/labour
contributions, completion, grid integration, and manager queries.
"""

import pytest

from core.economy.construction import (
    BuildPhase,
    Blueprint,
    ConstructionSite,
    ConstructionManager,
    DEFAULT_BLUEPRINTS,
    phase_for_progress,
)
from core.npc.models import NPC, PersonalityTraits
from core.world.grid import Grid, WorldObject


# ---------- Helpers ----------


def _make_npc(
    npc_id: str = "npc_1",
    name: str = "Aldric",
    occupation: str = "labourer",
    gold: int = 50,
    inventory: dict | None = None,
) -> NPC:
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory="A test NPC.",
        occupation=occupation,
        gold=gold,
        inventory=inventory if inventory is not None else {},
    )


def _make_blueprint(
    blueprint_id: str = "test_hut",
    width: int = 2,
    height: int = 2,
    resources: dict | None = None,
    labour: float = 30.0,
) -> Blueprint:
    return Blueprint(
        blueprint_id=blueprint_id,
        building_type="test_hut",
        name="Test Hut",
        width=width,
        height=height,
        required_resources=resources if resources is not None else {"wood": 10, "stone": 5},
        labour_required=labour,
    )


def _make_site(
    blueprint: Blueprint | None = None,
    x: int = 5,
    z: int = 5,
) -> ConstructionSite:
    bp = blueprint or _make_blueprint()
    return ConstructionSite(
        site_id="site_test",
        blueprint=bp,
        x=x, z=z,
    )


def _make_grid(width: int = 30, height: int = 30) -> Grid:
    return Grid(width, height)


# ---------- BuildPhase ----------


class TestBuildPhase:

    def test_phase_for_zero(self):
        assert phase_for_progress(0.0) == BuildPhase.PLANNED

    def test_phase_for_quarter(self):
        assert phase_for_progress(0.25) == BuildPhase.FOUNDATION

    def test_phase_for_half(self):
        assert phase_for_progress(0.50) == BuildPhase.WALLS

    def test_phase_for_three_quarters(self):
        assert phase_for_progress(0.75) == BuildPhase.ROOFING

    def test_phase_for_complete(self):
        assert phase_for_progress(1.0) == BuildPhase.COMPLETE

    def test_phase_intermediate(self):
        """Between thresholds, stays at the lower phase."""
        assert phase_for_progress(0.10) == BuildPhase.PLANNED
        assert phase_for_progress(0.40) == BuildPhase.FOUNDATION
        assert phase_for_progress(0.60) == BuildPhase.WALLS
        assert phase_for_progress(0.90) == BuildPhase.ROOFING


# ---------- Blueprint ----------


class TestBlueprint:

    def test_total_resource_units(self):
        bp = _make_blueprint(resources={"wood": 10, "stone": 5})
        assert bp.total_resource_units == 15

    def test_total_resource_units_empty(self):
        bp = _make_blueprint(resources={})
        assert bp.total_resource_units == 0

    def test_to_dict(self):
        bp = _make_blueprint()
        d = bp.to_dict()
        assert d["blueprint_id"] == "test_hut"
        assert d["width"] == 2
        assert "wood" in d["required_resources"]

    def test_default_blueprints_exist(self):
        assert "church" in DEFAULT_BLUEPRINTS
        assert "home" in DEFAULT_BLUEPRINTS
        assert "watchtower" in DEFAULT_BLUEPRINTS


# ---------- ConstructionSite — Progress ----------


class TestConstructionSiteProgress:

    def test_initial_progress_zero(self):
        site = _make_site()
        assert site.progress == 0.0
        assert site.phase == BuildPhase.PLANNED
        assert not site.is_complete

    def test_resource_progress(self):
        site = _make_site()  # needs wood:10, stone:5 = 15 total
        site.contributed["wood"] = 10  # 10 of 15
        site.contributed["stone"] = 0
        assert site.resource_progress == pytest.approx(10 / 15)

    def test_resource_progress_full(self):
        site = _make_site()
        site.contributed["wood"] = 10
        site.contributed["stone"] = 5
        assert site.resource_progress == pytest.approx(1.0)

    def test_labour_progress(self):
        site = _make_site()  # labour_required = 30
        site.labour_contributed = 15.0
        assert site.labour_progress == pytest.approx(0.5)

    def test_overall_progress_weighted(self):
        """70% resources + 30% labour."""
        site = _make_site()
        site.contributed["wood"] = 10
        site.contributed["stone"] = 5  # resource 100%
        site.labour_contributed = 15.0  # labour 50%
        expected = 1.0 * 0.7 + 0.5 * 0.3
        assert site.progress == pytest.approx(expected)

    def test_is_complete(self):
        site = _make_site()
        site.contributed["wood"] = 10
        site.contributed["stone"] = 5
        site.labour_contributed = 30.0
        assert site.is_complete

    def test_not_complete_missing_labour(self):
        site = _make_site()
        site.contributed["wood"] = 10
        site.contributed["stone"] = 5
        site.labour_contributed = 0.0
        assert not site.is_complete

    def test_resources_still_needed(self):
        site = _make_site()
        site.contributed["wood"] = 6
        needed = site.resources_still_needed()
        assert needed["wood"] == 4
        assert needed["stone"] == 5

    def test_resources_still_needed_all_met(self):
        site = _make_site()
        site.contributed["wood"] = 10
        site.contributed["stone"] = 5
        assert site.resources_still_needed() == {}

    def test_no_labour_blueprint(self):
        bp = _make_blueprint(labour=0.0)
        site = _make_site(blueprint=bp)
        site.contributed["wood"] = 10
        site.contributed["stone"] = 5
        assert site.is_complete


# ---------- ConstructionSite — Contributions ----------


class TestContributions:

    def test_contribute_resources(self):
        site = _make_site()
        npc = _make_npc(inventory={"wood": 20})
        accepted, msg = site.contribute_resources(npc, "wood", 5)
        assert accepted == 5
        assert msg == "ok"
        assert site.contributed["wood"] == 5
        assert npc.inventory["wood"] == 15

    def test_contribute_caps_at_needed(self):
        site = _make_site()  # needs 10 wood
        npc = _make_npc(inventory={"wood": 50})
        accepted, _ = site.contribute_resources(npc, "wood", 50)
        assert accepted == 10
        assert site.contributed["wood"] == 10
        assert npc.inventory["wood"] == 40

    def test_contribute_unneeded_resource(self):
        site = _make_site()
        npc = _make_npc(inventory={"iron": 10})
        accepted, msg = site.contribute_resources(npc, "iron", 5)
        assert accepted == 0
        assert "not needed" in msg

    def test_contribute_no_inventory(self):
        site = _make_site()
        npc = _make_npc(inventory={})
        accepted, msg = site.contribute_resources(npc, "wood", 5)
        assert accepted == 0
        assert "no wood" in msg

    def test_contribute_already_met(self):
        site = _make_site()
        site.contributed["wood"] = 10
        npc = _make_npc(inventory={"wood": 5})
        accepted, msg = site.contribute_resources(npc, "wood", 5)
        assert accepted == 0
        assert "already met" in msg

    def test_contributor_tracking(self):
        site = _make_site()
        npc = _make_npc(npc_id="builder_1", inventory={"wood": 20})
        site.contribute_resources(npc, "wood", 3)
        site.contribute_resources(npc, "wood", 2)
        assert site.contributors["builder_1"] == 5

    def test_contribute_labour(self):
        site = _make_site()
        site.contributed["wood"] = 5  # need some resources first
        npc = _make_npc()
        accepted, msg = site.contribute_labour(npc, 10.0)
        assert accepted == 10.0
        assert msg == "ok"
        assert site.labour_contributed == 10.0

    def test_contribute_labour_caps_at_needed(self):
        site = _make_site()  # labour_required = 30
        site.contributed["wood"] = 5
        npc = _make_npc()
        accepted, _ = site.contribute_labour(npc, 100.0)
        assert accepted == 30.0
        assert site.labour_contributed == 30.0

    def test_contribute_labour_no_resources_yet(self):
        site = _make_site()
        npc = _make_npc()
        accepted, msg = site.contribute_labour(npc, 10.0)
        assert accepted == 0.0
        assert "no resources" in msg

    def test_summary(self):
        site = _make_site()
        site.contributed["wood"] = 5
        s = site.summary()
        assert "Test Hut" in s
        assert "wood" in s

    def test_to_dict(self):
        site = _make_site()
        d = site.to_dict()
        assert d["site_id"] == "site_test"
        assert d["phase"] == "planned"
        assert "resources_needed" in d


# ---------- ConstructionManager — Site Creation ----------


class TestManagerCreation:

    def test_start_construction(self):
        grid = _make_grid()
        mgr = ConstructionManager()
        site, msg = mgr.start_construction("home", 5, 5, grid)
        assert site is not None
        assert msg == "ok"
        assert site.phase == BuildPhase.PLANNED

    def test_start_places_object_on_grid(self):
        grid = _make_grid()
        mgr = ConstructionManager()
        site, _ = mgr.start_construction("home", 5, 5, grid)
        tile = grid.get_tile(5, 5)
        assert any(o.object_type == "construction" for o in tile.objects)

    def test_start_marks_footprint_unwalkable(self):
        grid = _make_grid()
        mgr = ConstructionManager()
        bp = DEFAULT_BLUEPRINTS["home"]  # 2x3
        site, _ = mgr.start_construction("home", 5, 5, grid)
        # (5,5) stays walkable (access tile), others should not
        assert grid.get_tile(5, 5).walkable  # access tile
        assert not grid.get_tile(6, 5).walkable
        assert not grid.get_tile(5, 6).walkable

    def test_start_unknown_blueprint(self):
        grid = _make_grid()
        mgr = ConstructionManager()
        site, msg = mgr.start_construction("flying_castle", 5, 5, grid)
        assert site is None
        assert "unknown" in msg

    def test_start_out_of_bounds(self):
        grid = _make_grid()
        mgr = ConstructionManager()
        site, msg = mgr.start_construction("home", 99, 99, grid)
        assert site is None
        assert "out of bounds" in msg

    def test_start_occupied_tile(self):
        grid = _make_grid()
        grid.place_object(5, 5, WorldObject(
            object_id="rock", object_type="resource", name="Rock",
        ))
        mgr = ConstructionManager()
        site, msg = mgr.start_construction("home", 5, 5, grid)
        assert site is None
        assert "occupied" in msg

    def test_start_impassable_tile(self):
        grid = _make_grid()
        tile = grid.get_tile(5, 5)
        tile.walkable = False
        mgr = ConstructionManager()
        site, msg = mgr.start_construction("home", 5, 5, grid)
        assert site is None
        assert "not passable" in msg

    def test_custom_blueprint(self):
        grid = _make_grid()
        bp = _make_blueprint(blueprint_id="shed", width=2, height=2)
        mgr = ConstructionManager()
        mgr.add_blueprint(bp)
        site, msg = mgr.start_construction("shed", 5, 5, grid)
        assert site is not None


# ---------- ConstructionManager — Contributions ----------


class TestManagerContributions:

    def setup_method(self):
        self.grid = _make_grid()
        self.mgr = ConstructionManager()
        bp = _make_blueprint()
        self.mgr.add_blueprint(bp)
        self.site, _ = self.mgr.start_construction("test_hut", 5, 5, self.grid)
        self.npc = _make_npc(inventory={"wood": 50, "stone": 30})

    def test_contribute_resources_via_manager(self):
        accepted, msg = self.mgr.contribute_resources(
            self.npc, self.site.site_id, "wood", 5,
        )
        assert accepted == 5
        assert msg == "ok"

    def test_contribute_to_nonexistent_site(self):
        accepted, msg = self.mgr.contribute_resources(
            self.npc, "fake_site", "wood", 5,
        )
        assert accepted == 0
        assert "not found" in msg

    def test_contribute_labour_via_manager(self):
        # Need some resources first
        self.mgr.contribute_resources(self.npc, self.site.site_id, "wood", 5)
        accepted, msg = self.mgr.contribute_labour(
            self.npc, self.site.site_id, 10.0,
        )
        assert accepted == 10.0

    def test_contribute_labour_nonexistent_site(self):
        accepted, msg = self.mgr.contribute_labour(self.npc, "fake", 10.0)
        assert accepted == 0.0


# ---------- ConstructionManager — Completion ----------


class TestManagerCompletion:

    def setup_method(self):
        self.grid = _make_grid()
        self.events = []
        self.mgr = ConstructionManager(
            on_event=lambda t, p, d: self.events.append((t, p, d)),
        )
        bp = _make_blueprint()
        self.mgr.add_blueprint(bp)
        self.site, _ = self.mgr.start_construction("test_hut", 5, 5, self.grid)
        self.npc = _make_npc(inventory={"wood": 50, "stone": 30})

    def _complete_site(self):
        self.mgr.contribute_resources(self.npc, self.site.site_id, "wood", 10)
        self.mgr.contribute_resources(self.npc, self.site.site_id, "stone", 5)
        self.mgr.contribute_labour(self.npc, self.site.site_id, 30.0)

    def test_check_incomplete(self):
        ok, msg = self.mgr.check_and_complete(self.site.site_id, self.grid)
        assert not ok
        assert "not complete" in msg

    def test_check_and_complete(self):
        self._complete_site()
        ok, msg = self.mgr.check_and_complete(self.site.site_id, self.grid)
        assert ok
        assert msg == "ok"

    def test_complete_places_building(self):
        self._complete_site()
        self.mgr.check_and_complete(self.site.site_id, self.grid)
        tile = self.grid.get_tile(5, 5)
        building_objs = [o for o in tile.objects if o.object_type == "building"]
        assert len(building_objs) == 1
        assert building_objs[0].name == "Test Hut"

    def test_complete_removes_construction_object(self):
        self._complete_site()
        self.mgr.check_and_complete(self.site.site_id, self.grid)
        tile = self.grid.get_tile(5, 5)
        construction_objs = [o for o in tile.objects if o.object_type == "construction"]
        assert len(construction_objs) == 0

    def test_complete_marks_footprint_unwalkable(self):
        self._complete_site()
        self.mgr.check_and_complete(self.site.site_id, self.grid)
        for dx in range(2):
            for dz in range(2):
                assert not self.grid.get_tile(5 + dx, 5 + dz).walkable

    def test_complete_fires_event(self):
        self._complete_site()
        self.mgr.check_and_complete(self.site.site_id, self.grid)
        assert len(self.events) == 1
        assert self.events[0][0] == "construction_complete"
        assert self.npc.npc_id in self.events[0][1]

    def test_complete_removes_from_active(self):
        self._complete_site()
        self.mgr.check_and_complete(self.site.site_id, self.grid)
        assert self.mgr.get_site(self.site.site_id) is None

    def test_complete_nonexistent(self):
        ok, msg = self.mgr.check_and_complete("fake", self.grid)
        assert not ok


# ---------- ConstructionManager — Queries ----------


class TestManagerQueries:

    def setup_method(self):
        self.grid = _make_grid()
        self.mgr = ConstructionManager()
        bp1 = _make_blueprint(blueprint_id="hut_a", resources={"wood": 10})
        bp2 = _make_blueprint(blueprint_id="hut_b", resources={"stone": 10})
        self.mgr.add_blueprint(bp1)
        self.mgr.add_blueprint(bp2)
        self.site_a, _ = self.mgr.start_construction("hut_a", 2, 2, self.grid)
        self.site_b, _ = self.mgr.start_construction("hut_b", 8, 8, self.grid)

    def test_get_all_sites(self):
        assert len(self.mgr.get_all_sites()) == 2

    def test_get_sites_needing_wood(self):
        sites = self.mgr.get_sites_needing("wood")
        assert len(sites) == 1
        assert sites[0].site_id == self.site_a.site_id

    def test_get_sites_needing_stone(self):
        sites = self.mgr.get_sites_needing("stone")
        assert len(sites) == 1
        assert sites[0].site_id == self.site_b.site_id

    def test_get_nearest_site(self):
        nearest = self.mgr.get_nearest_site(1, 1)
        assert nearest.site_id == self.site_a.site_id

    def test_get_nearest_site_filtered(self):
        nearest = self.mgr.get_nearest_site(1, 1, resource="stone")
        assert nearest.site_id == self.site_b.site_id

    def test_get_nearest_site_none(self):
        assert self.mgr.get_nearest_site(0, 0, resource="iron") is None


# ---------- ConstructionManager — Evaluate Contribution ----------


class TestEvaluateContribution:

    def test_labourer_always_willing(self):
        site = _make_site()
        npc = _make_npc(occupation="labourer")
        willing, _ = ConstructionManager().evaluate_contribution(npc, site)
        assert willing

    def test_npc_with_resources_willing(self):
        site = _make_site()
        npc = _make_npc(occupation="merchant", inventory={"wood": 10})
        willing, _ = ConstructionManager().evaluate_contribution(npc, site)
        assert willing

    def test_far_npc_unwilling(self):
        site = _make_site(x=0, z=0)
        npc = _make_npc(occupation="farmer")
        npc.x = 50
        npc.z = 50
        willing, reason = ConstructionManager().evaluate_contribution(npc, site)
        assert not willing
        assert "far" in reason


# ---------- ConstructionManager — State ----------


class TestManagerState:

    def test_get_state(self):
        grid = _make_grid()
        mgr = ConstructionManager()
        mgr.start_construction("home", 5, 5, grid)
        state = mgr.get_state()
        assert len(state["active_sites"]) == 1
        assert state["completed_count"] == 0

    def test_get_stats(self):
        mgr = ConstructionManager()
        stats = mgr.get_stats()
        assert stats["active_sites"] == 0
        assert stats["blueprints_available"] == len(DEFAULT_BLUEPRINTS)
