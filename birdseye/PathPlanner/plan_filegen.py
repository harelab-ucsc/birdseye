#!/usr/bin/env python3
"""
@file   plan.py

Utilities for working with QGroundControl-formatted flight plans (.plan).
Adapted from KML generator script. Auto-generates a bounding geofence.

@usage  python3 plan.py input.csv -o flight.plan
"""

import argparse
import csv
import sys
import json
from geopy.distance import distance as geodist
from typing import Union

# ==========================================
# CONFIGURATION
# ==========================================

# Geofence padding distance in meters
GEOFENCE_PADDING_M: float = 5.0

# MAVLink Firmware Type (MAV_AUTOPILOT enum)
# Docs: https://mavlink.io/en/messages/common.html#MAV_AUTOPILOT
# 12 = MAV_AUTOPILOT_PX4
FIRMWARE_TYPE: int = 12

# MAVLink Vehicle Type (MAV_TYPE enum)
# Docs: https://mavlink.io/en/messages/common.html#MAV_TYPE
# 2 = MAV_TYPE_QUADROTOR
VEHICLE_TYPE: int = 2

# Other flight plan defaults
DEFAULT_PLAN_PATH = "./flight.plan"
DEFAULT_ACTIONS_SEQUENCE: Union[str, None] = None
DEFAULT_GIMBAL: Union[float, None] = None
DEFAULT_HEADING: Union[float, None] = None
DEFAULT_HEIGHT: float = 20.0  # m
DEFAULT_SPEED: float = 2.3  # m/s
DEFAULT_TURNMODE: str = "AUTO"

# ==========================================


def _write_file(path: str, data: str):
    with open(path, "w+") as fp:
        fp.write(data)


def _read_csv_file(path: str) -> list[dict]:
    """Parse the CSV, automatically handling headers if present."""
    data = []
    with open(path, newline="") as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)

        has_header = False
        try:
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            pass

        if has_header or "lat" in sample.splitlines()[0].lower():
            csv_lines = csv.DictReader(csvfile)
        else:
            # Fallback for headless CSVs (includes extended fields from original script)
            fieldnames = [
                "lat",
                "lon",
                "altWGS84",
                "altMSL",
                "gpsStatus",
                "label",
                "speed",
                "heading",
                "turnmode",
                "actions_sequence",
                "gimbal",
            ]
            csv_lines = csv.DictReader(csvfile, fieldnames=fieldnames)

        for row in csv_lines:
            # Skip manual header row if it slipped through
            if row.get("lat") in ("lat", "_lat", "Latitude"):
                continue
            data.append(row)
    return data


def make_simple_item(
    command, params, lat=0.0, lon=0.0, alt=0.0, frame=3, autoContinue=True
):
    """Helper to generate QGroundControl Mission Items (MAVLink commands)."""
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": float(alt),
        "AltitudeMode": 1,
        "autoContinue": autoContinue,
        "command": command,
        "doJumpId": 1,
        "frame": frame,  # 3 = Global Relative Alt, 2 = Mission
        "params": params,
        "type": "SimpleItem",
    }


def plan_2_qgc(plan: list[dict], on_finish: str = "Hover", wait_ms: int = 0) -> str:
    items = []
    lats = []
    lons = []
    planned_home = None
    current_speed = None

    for row in plan:
        name = row.get("label", "Waypoint")

        # Parse coordinates safely
        lat_str = str(row.get("lat", ""))
        lon_str = str(row.get("lon", ""))
        if lat_str.startswith("_"):
            lat_str = lat_str[1:]
        if lon_str.startswith("_"):
            lon_str = lon_str[1:]

        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            continue

        lats.append(lat)
        lons.append(lon)

        # Parse extra fields with fallbacks
        alt_wgs84 = row.get("altWGS84")
        height = float(alt_wgs84) + 10.0 if alt_wgs84 else DEFAULT_HEIGHT

        raw_speed = row.get("speed")
        speed = float(raw_speed) if raw_speed else DEFAULT_SPEED

        heading = row.get("heading") or DEFAULT_HEADING
        heading = float(heading) if heading is not None else None

        actions_sequence = row.get("actions_sequence") or DEFAULT_ACTIONS_SEQUENCE
        turnmode = row.get("turnmode") or DEFAULT_TURNMODE

        if speed > 15 or speed <= 0:
            sys.exit(f"Speed should be >0 or <=15 m/s for {name}")

        if not planned_home:
            planned_home = [lat, lon, height]

        # 1. Update Vehicle Speed (MAV_CMD_DO_CHANGE_SPEED = 178)
        if speed != current_speed:
            items.append(make_simple_item(178, [1, speed, -1, 0, 0, 0, 0], frame=2))
            current_speed = speed

        # 2. Add Navigational Waypoint (MAV_CMD_NAV_WAYPOINT = 16)
        yaw_param = heading if heading is not None else None
        items.append(
            make_simple_item(
                16, [0, 0, 0, yaw_param, lat, lon, height], lat=lat, lon=lon, alt=height
            )
        )

        # 3. Optional wait at each waypoint
        if wait_ms >= 500:
            items.append(
                make_simple_item(112, [wait_ms / 1000.0, 0, 0, 0, 0, 0, 0], frame=2)
            )

        # 4. Handle Sequence Actions
        if actions_sequence:
            for action in actions_sequence.split("."):
                action = action.upper().strip()
                if action == "SHOOT":
                    items.append(make_simple_item(2000, [0, 0, 1, 0, 0, 0, 0], frame=2))
                elif action == "REC":
                    items.append(make_simple_item(2500, [0, 0, 0, 0, 0, 0, 0], frame=2))
                elif action == "STOPREC":
                    items.append(make_simple_item(2501, [0, 0, 0, 0, 0, 0, 0], frame=2))
                elif action.startswith("G"):
                    g_angle = float(action[1:])
                    # MAV_CMD_DO_MOUNT_CONTROL = 205 (Gimbal pitch target)
                    items.append(
                        make_simple_item(205, [g_angle, 0, 0, 0, 0, 0, 2], frame=2)
                    )
                elif action.startswith("A"):
                    a_yaw = float(action[1:])
                    # MAV_CMD_CONDITION_YAW = 115
                    dir_flag = (
                        1
                        if turnmode in ("C", "CLOCKWISE")
                        else -1
                        if turnmode in ("CC", "COUNTERCLOCKWISE")
                        else 0
                    )
                    items.append(
                        make_simple_item(115, [a_yaw, 0, dir_flag, 0, 0, 0, 0], frame=2)
                    )
                elif action.startswith("H"):
                    delay_ms = float(action[1:])
                    if delay_ms < 500:
                        sys.exit(
                            f"Hover length is in ms and should be >= 500 for {name}"
                        )
                    # MAV_CMD_CONDITION_DELAY = 112
                    items.append(
                        make_simple_item(
                            112, [delay_ms / 1000.0, 0, 0, 0, 0, 0, 0], frame=2
                        )
                    )

    # Optional Action: GoHome mapping
    if on_finish == "GoHome":
        # MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
        items.append(make_simple_item(20, [0, 0, 0, 0, 0, 0, 0], frame=2))

    # --- Geofence Generation ---
    geofence_dict = {"circles": [], "polygons": [], "version": 2}

    if lats and lons:
        max_lat, min_lat = max(lats), min(lats)
        max_lon, min_lon = max(lons), min(lons)

        # walk GEOFENCE_PADDING_M meters north/east from the mission's average latitude.
        avg_lat = (max_lat + min_lat) / 2.0
        _ref = (avg_lat, 0.0)
        lat_margin = (
            geodist(meters=GEOFENCE_PADDING_M).destination(_ref, bearing=0).latitude
            - avg_lat
        )
        lon_margin = (
            geodist(meters=GEOFENCE_PADDING_M).destination(_ref, bearing=90).longitude
        )

        # QGC requires a clockwise winding order for polygons
        polygon = [
            [max_lat + lat_margin, min_lon - lon_margin],  # Top Left
            [max_lat + lat_margin, max_lon + lon_margin],  # Top Right
            [min_lat - lat_margin, max_lon + lon_margin],  # Bottom Right
            [min_lat - lat_margin, min_lon - lon_margin],  # Bottom Left
        ]

        geofence_dict["polygons"].append(
            {"inclusion": True, "polygon": polygon, "version": 1}
        )

    # Assemble QGC Plan JSON object
    plan_json = {
        "fileType": "Plan",
        "geoFence": geofence_dict,
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": DEFAULT_SPEED,
            "firmwareType": FIRMWARE_TYPE,
            "hoverSpeed": 5.0,
            "items": items,
            "plannedHomePosition": planned_home or [0.0, 0.0, 0.0],
            "vehicleType": VEHICLE_TYPE,
            "version": 2,
        },
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }

    return json.dumps(plan_json, indent=4)


def csv2qgc(data_path: str, plan_path: str, on_finish: str = "GoHome", wait_ms: int = 0):
    # Convert an existing CSV path with flight data into a QGC .plan path
    flight_plan_csv = _read_csv_file(data_path)
    plan_str = plan_2_qgc(flight_plan_csv, on_finish, wait_ms=wait_ms)
    _write_file(path=plan_path, data=plan_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csvfile", type=argparse.FileType("r"), help="Specify csv input file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=DEFAULT_PLAN_PATH,
        help="Specify output file (default: ./flight.plan)",
    )
    parser.add_argument(
        "--onfinish",
        default="gohome",
        choices=["hover", "gohome"],
        help="Aircraft action when finish. hover or gohome (default: %(default)s).",
    )
    parser.add_argument(
        "--wait",
        default=0,
        type=int,
        metavar="MS",
        help="Hover duration in ms at each waypoint (min 500, default: disabled).",
    )
    args = parser.parse_args()

    on_finish = "Hover" if args.onfinish == "hover" else "GoHome"
    csv2qgc(args.csvfile.name, args.output.name, on_finish=on_finish, wait_ms=args.wait)
