#!/usr/bin/env python3
import sys
import os
import xml.etree.ElementTree as ET
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "birdseye"))
from PathPlanner.kml_filegen import plan_2_kml

KML_NS = "http://www.opengis.net/kml/2.2"
MIS_NS = "www.dji.com"


SINGLE_WP = [
    {
        "lat": "36.9741",
        "lon": "-122.0308",
        "altWGS84": "50.0",
        "label": "WP1",
        "turnmode": "AUTO",
        "actions_sequence": "SHOOT",
    }
]

TWO_WP = [
    {
        "lat": "36.9741",
        "lon": "-122.0308",
        "altWGS84": "50.0",
        "label": "WP1",
        "turnmode": "AUTO",
        "actions_sequence": "SHOOT",
    },
    {
        "lat": "36.9751",
        "lon": "-122.0298",
        "altWGS84": "60.0",
        "label": "WP2",
        "turnmode": "AUTO",
        "actions_sequence": "SHOOT",
    },
]


def _kml(rows, **kwargs):
    return plan_2_kml(rows, **kwargs)


def _parse(rows, **kwargs):
    return ET.fromstring(_kml(rows, **kwargs))


# --- XML validity ---


def test_output_is_valid_xml():
    root = _parse(SINGLE_WP)
    assert root is not None


def test_root_tag_is_kml():
    root = _parse(SINGLE_WP)
    assert root.tag == f"{{{KML_NS}}}kml" or root.tag == "kml"


# --- Waypoint placemarks ---


def test_waypoint_name_in_output():
    out = _kml(SINGLE_WP)
    assert "Waypoint1" in out


def test_waypoint_numbering_increments():
    out = _kml(TWO_WP)
    assert "Waypoint1" in out
    assert "Waypoint2" in out


def test_coordinates_present():
    out = _kml(SINGLE_WP)
    # lon,lat,height format
    assert "-122.0308" in out
    assert "36.9741" in out


def test_height_derived_from_altWGS84():
    out = _kml(SINGLE_WP)
    # altWGS84=50.0 + 10.0 = 60.0
    assert "60.0" in out


def test_all_waypoint_coords_in_wayline():
    out = _kml(TWO_WP)
    # Both lon values should appear in the LineString coordinates
    assert "-122.0308" in out
    assert "-122.0298" in out


# --- on_finish ---


def test_on_finish_hover():
    out = _kml(SINGLE_WP, on_finish="Hover")
    assert "<mis:actionOnFinish>Hover</mis:actionOnFinish>" in out


def test_on_finish_gohome():
    out = _kml(SINGLE_WP, on_finish="GoHome")
    assert "<mis:actionOnFinish>GoHome</mis:actionOnFinish>" in out


# --- turnmode ---

# turnMode only appears in the heading template, so these rows need heading+gimbal.
WP_WITH_HEADING = {**SINGLE_WP[0], "heading": "90", "gimbal": "-30.0"}


def test_turnmode_auto_normalized():
    out = _kml([{**WP_WITH_HEADING, "turnmode": "AUTO"}])
    assert "<mis:turnMode>Auto</mis:turnMode>" in out


def test_turnmode_clockwise():
    out = _kml([{**WP_WITH_HEADING, "turnmode": "C"}])
    assert "<mis:turnMode>Clockwise</mis:turnMode>" in out


def test_turnmode_counterclockwise():
    out = _kml([{**WP_WITH_HEADING, "turnmode": "CC"}])
    assert "<mis:turnMode>Counterclockwise</mis:turnMode>" in out


# --- Actions ---


def test_action_shoot():
    out = _kml(SINGLE_WP)
    assert "ShootPhoto" in out


def test_action_record():
    row = {**SINGLE_WP[0], "actions_sequence": "REC"}
    out = _kml([row])
    assert "StartRecording" in out


def test_action_stoprec():
    row = {**SINGLE_WP[0], "actions_sequence": "STOPREC"}
    out = _kml([row])
    assert "StopRecording" in out


def test_action_gimbal():
    row = {**SINGLE_WP[0], "actions_sequence": "G-30"}
    out = _kml([row])
    assert "GimbalPitch" in out
    assert 'param="-30"' in out


def test_action_aircraft_yaw():
    row = {**SINGLE_WP[0], "actions_sequence": "A90"}
    out = _kml([row])
    assert "AircraftYaw" in out
    assert 'param="90"' in out


def test_action_hover_delay():
    row = {**SINGLE_WP[0], "actions_sequence": "H1000"}
    out = _kml([row])
    assert "Hovering" in out
    assert 'param="1000"' in out


def test_action_sequence_multiple():
    row = {**SINGLE_WP[0], "actions_sequence": "SHOOT.REC"}
    out = _kml([row])
    assert "ShootPhoto" in out
    assert "StartRecording" in out


# --- Heading ---


def test_no_heading_uses_wayline_altitude():
    # Without heading, the no-heading template sets useWaylineAltitude=true
    out = _kml(SINGLE_WP)
    assert "<mis:useWaylineAltitude>true</mis:useWaylineAltitude>" in out


def test_heading_uses_point_altitude():
    row = {**SINGLE_WP[0], "heading": "90", "gimbal": "-30.0"}
    out = _kml([row])
    assert "<mis:useWaylineAltitude>false</mis:useWaylineAltitude>" in out
    assert "<mis:heading>90</mis:heading>" in out
