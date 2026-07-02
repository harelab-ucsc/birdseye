#!/usr/bin/env python3
import json
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "birdseye"))
from PathPlanner.plan_filegen import plan_2_qgc, make_simple_item, GEOFENCE_PADDING_M

# Minimal single-waypoint plan row
SINGLE_WP = [{"lat": "36.9741", "lon": "-122.0308", "altWGS84": "50.0", "speed": "5.0"}]

# Two-waypoint plan spanning a small area
TWO_WP = [
    {"lat": "36.9741", "lon": "-122.0308", "altWGS84": "50.0", "speed": "5.0"},
    {"lat": "36.9751", "lon": "-122.0298", "altWGS84": "50.0", "speed": "5.0"},
]


def _parse(plan_rows, **kwargs):
    return json.loads(plan_2_qgc(plan_rows, **kwargs))


# --- Output structure ---


def test_output_is_valid_json():
    result = plan_2_qgc(SINGLE_WP)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


def test_top_level_keys():
    d = _parse(SINGLE_WP)
    assert d["fileType"] == "Plan"
    assert d["groundStation"] == "QGroundControl"
    assert "mission" in d
    assert "geoFence" in d


# --- Mission items ---


def test_waypoint_item_present():
    d = _parse(SINGLE_WP)
    items = d["mission"]["items"]
    nav_wps = [i for i in items if i.get("command") == 16]
    assert len(nav_wps) == 1


def test_waypoint_coordinates():
    d = _parse(SINGLE_WP)
    items = d["mission"]["items"]
    wp = next(i for i in items if i.get("command") == 16)
    assert wp["params"][4] == pytest.approx(36.9741)
    assert wp["params"][5] == pytest.approx(-122.0308)


def test_speed_item_emitted():
    d = _parse(SINGLE_WP)
    items = d["mission"]["items"]
    speed_items = [i for i in items if i.get("command") == 178]
    assert len(speed_items) == 1
    assert speed_items[0]["params"][1] == pytest.approx(5.0)


def test_speed_item_not_duplicated_when_constant():
    """Only one speed command when all waypoints share the same speed."""
    d = _parse(TWO_WP)
    speed_items = [i for i in d["mission"]["items"] if i.get("command") == 178]
    assert len(speed_items) == 1


def test_speed_item_emitted_per_change():
    plan = [
        {"lat": "36.9741", "lon": "-122.0308", "speed": "3.0"},
        {"lat": "36.9751", "lon": "-122.0298", "speed": "5.0"},
    ]
    d = _parse(plan)
    speed_items = [i for i in d["mission"]["items"] if i.get("command") == 178]
    assert len(speed_items) == 2


def test_planned_home_is_first_waypoint():
    d = _parse(TWO_WP)
    home = d["mission"]["plannedHomePosition"]
    assert home[0] == pytest.approx(36.9741)
    assert home[1] == pytest.approx(-122.0308)


def test_two_waypoints_produce_two_nav_items():
    d = _parse(TWO_WP)
    nav_wps = [i for i in d["mission"]["items"] if i.get("command") == 16]
    assert len(nav_wps) == 2


def test_gohome_appends_rtl():
    d = _parse(SINGLE_WP, on_finish="GoHome")
    items = d["mission"]["items"]
    assert items[-1]["command"] == 20


def test_hover_does_not_append_rtl():
    d = _parse(SINGLE_WP, on_finish="Hover")
    items = d["mission"]["items"]
    assert all(i["command"] != 20 for i in items)


# --- Actions sequence ---


def test_action_shoot():
    plan = [
        {
            "lat": "36.9741",
            "lon": "-122.0308",
            "speed": "3.0",
            "actions_sequence": "SHOOT",
        }
    ]
    d = _parse(plan)
    assert any(i["command"] == 2000 for i in d["mission"]["items"])


def test_action_hover_delay():
    plan = [
        {
            "lat": "36.9741",
            "lon": "-122.0308",
            "speed": "3.0",
            "actions_sequence": "H1000",
        }
    ]
    d = _parse(plan)
    delay_items = [i for i in d["mission"]["items"] if i["command"] == 112]
    assert len(delay_items) == 1
    assert delay_items[0]["params"][0] == pytest.approx(1.0)


# --- Geofence ---


def test_geofence_polygon_generated():
    d = _parse(TWO_WP)
    polygons = d["geoFence"]["polygons"]
    assert len(polygons) == 1
    assert polygons[0]["inclusion"] is True
    assert polygons[0]["version"] == 1
    assert len(polygons[0]["polygon"]) == 4


def test_geofence_encloses_waypoints():
    d = _parse(TWO_WP)
    poly = d["geoFence"]["polygons"][0]["polygon"]
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    for wp in TWO_WP:
        assert min(lats) < float(wp["lat"]) < max(lats)
        assert min(lons) < float(wp["lon"]) < max(lons)


def test_geofence_padding_at_least_declared_distance():
    """Each polygon edge must be >= GEOFENCE_PADDING_M from the nearest waypoint."""
    from geopy.distance import distance as geodist

    d = _parse(TWO_WP)
    poly = d["geoFence"]["polygons"][0]["polygon"]
    wp_lats = [float(w["lat"]) for w in TWO_WP]
    wp_lons = [float(w["lon"]) for w in TWO_WP]

    max_lat, min_lat = max(wp_lats), min(wp_lats)
    max_lon, min_lon = max(wp_lons), min(wp_lons)
    poly_max_lat = max(p[0] for p in poly)
    poly_min_lat = min(p[0] for p in poly)
    poly_max_lon = max(p[1] for p in poly)
    poly_min_lon = min(p[1] for p in poly)

    avg_lat = (max_lat + min_lat) / 2.0
    north_pad = geodist((max_lat, avg_lat), (poly_max_lat, avg_lat)).meters
    south_pad = geodist((min_lat, avg_lat), (poly_min_lat, avg_lat)).meters
    east_pad = geodist((avg_lat, max_lon), (avg_lat, poly_max_lon)).meters
    west_pad = geodist((avg_lat, min_lon), (avg_lat, poly_min_lon)).meters

    for pad in (north_pad, south_pad, east_pad, west_pad):
        assert pad == pytest.approx(GEOFENCE_PADDING_M, rel=0.01)


def test_empty_plan_returns_valid_structure():
    d = _parse([])
    assert d["fileType"] == "Plan"
    assert d["geoFence"]["polygons"] == []
    assert d["mission"]["items"] == []


# --- make_simple_item helper ---


def test_make_simple_item_structure():
    item = make_simple_item(
        16, [0, 0, 0, None, 1.0, 2.0, 10.0], lat=1.0, lon=2.0, alt=10.0
    )
    assert item["command"] == 16
    assert item["type"] == "SimpleItem"
    assert item["Altitude"] == pytest.approx(10.0)
