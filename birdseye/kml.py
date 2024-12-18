#!/usr/bin/env python3
"""
@file   kml.py

Utilities for working with KML-formatted flight plans for the M300.
Adapated from [csv2djipilot]{https://github.com/IPGP/csv2djipilot/tree/main}.

@author HARE Lab
@date   17 Dec 2024
@todo
    * Set header if not supplied.
"""
from string import Template
from typing import Union

import argparse
import csv
import sys


DEFAULT_KML_PATH = "./catch/flight.kml"
#_CURRENT_ALTITUDE = 30
DEFAULT_ACTIONS_SEQUENCE: Union[str, None] = None
DEFAULT_GIMBAL: Union[float, None] = None
DEFAULT_HEADING: Union[float, None] = None
DEFAULT_HEIGHT: float = 10    # m
DEFAULT_SPEED: float = 2.3    # m/s
DEFAULT_TURNMODE: str = "AUTO"

#print(f"{CsvFile} to {args.output.name}")
#CsvFile = "exemple.csv"
#CsvFile = "exemple_simple.csv"
CSV_HEADER = False

def _write_file(path: str, data: any):
    with open(path, "w+") as fp:
        fp.write(data)

def _read_csv_file(path: str) -> list[dict]:
    """_read_csv_file(path) -> data

    Generate a list of CSV lines from input CSV file.
    
    @param  path (str)          Input path.
    @return data (list[dict])   Output data.
    """
    data = []
    with open(path, newline="") as csvfile:
        fieldnames = ["lat", "lon", "altWGS84", "altMSL", "gpsStatus", "label"]
        csv_lines = csv.DictReader(csvfile, fieldnames=fieldnames)
        for row in csv_lines:
            data.append(row)
    return data

def plan_2_kml(
    plan: list[dict],
    on_finish: str = "hover"
) -> str:
    """plan_2_kml(plan, hover) -> kml_str

    Generate a string-ified KML from a flight plan, ready for writing to a file.
    
    @param  plan (list[dict])   Path to be converted.
    @param  on_finish (str)     "hover" by default.
    """
    kml_str = """<?xml version="1.0" encoding="UTF-8"?>

    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document xmlns="">
        <name>chambon_small</name>
        <open>1</open>
        <ExtendedData xmlns:mis="www.dji.com">
          <mis:type>Waypoint</mis:type>
          <mis:stationType>0</mis:stationType>
        </ExtendedData>
        <Style id="waylineGreenPoly">
          <LineStyle>
            <color>FF0AEE8B</color>
            <width>6</width>
          </LineStyle>
        </Style>
        <Style id="waypointStyle">
          <IconStyle>
            <Icon>
              <href>https://cdnen.dji-flighthub.com/static/app/images/point.png</href>
            </Icon>
          </IconStyle>
        </Style>
        <Folder>
          <name>Waypoints</name>
          <description>Waypoints in the Mission.</description>\n"""
    all_coordinates = ""
    waypoint_number = 1

    waypoint_start = Template("""      <Placemark>
            <name>Waypoint$waypoint_number</name>
            <visibility>1</visibility>
            <description>Waypoint</description>
            <styleUrl>#waypointStyle</styleUrl>
            <ExtendedData xmlns:mis="www.dji.com">
              <mis:useWaylineAltitude>false</mis:useWaylineAltitude>
              <mis:heading>$heading</mis:heading>
              <mis:turnMode>$turnmode</mis:turnMode>
              <mis:gimbalPitch>$gimbal</mis:gimbalPitch>
              <mis:useWaylineSpeed>false</mis:useWaylineSpeed>
              <mis:speed>$speed</mis:speed>
              <mis:useWaylineHeadingMode>true</mis:useWaylineHeadingMode>
              <mis:useWaylinePointType>true</mis:useWaylinePointType>
              <mis:pointType>LineStop</mis:pointType>
              <mis:cornerRadius>0.2</mis:cornerRadius>""")

    waypoint_start_no_heading = Template("""      <Placemark>
            <name>Waypoint$waypoint_number</name>
            <visibility>1</visibility>
            <description>Waypoint</description>
            <styleUrl>#waypointStyle</styleUrl>
            <ExtendedData xmlns:mis="www.dji.com">
              <mis:useWaylineAltitude>true</mis:useWaylineAltitude>
              <mis:speed>2.3</mis:speed>#
              <mis:useWaylineHeadingMode>true</mis:useWaylineHeadingMode>
              <mis:useWaylinePointType>true</mis:useWaylinePointType>
              <mis:pointType>LineStop</mis:pointType>
              <mis:cornerRadius>0.2</mis:cornerRadius>""")

    waypoint_end = Template("""
            </ExtendedData>
            <Point>
              <altitudeMode>relativeToGround</altitudeMode>
              <coordinates>$lon,$lat,$height</coordinates>
            </Point>
          </Placemark>""")
    hover_template = Template("""
              <mis:actions param="$length" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">Hovering</mis:actions>""")
    shoot_template = Template("""
              <mis:actions param="0" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">ShootPhoto</mis:actions>""")

    gimbal_template = Template("""
              <mis:actions param="$gimbal_angle" accuracy="1" cameraIndex="0" payloadType="0" payloadIndex="0">GimbalPitch</mis:actions>""")
    aircraftyaw_template = Template("""
              <mis:actions param="$aircraftyaw" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">AircraftYaw</mis:actions>""")
    record_template = Template("""
              <mis:actions param="0" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">StartRecording</mis:actions>""")
    stoprecord_template = Template("""
              <mis:actions param="0" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">StopRecording</mis:actions>""")


    all_coordinates_template = Template("$lon,$lat,$height")
#        <mis:altitude>$_CURRENT_ALTITUDE</mis:altitude>
    xml_end = Template("""    </Folder>
        <Placemark>
          <name>Wayline</name>
          <description>Wayline</description>
          <visibility>1</visibility>
          <ExtendedData xmlns:mis="www.dji.com">
            <mis:autoFlightSpeed>2.3</mis:autoFlightSpeed>
            <mis:actionOnFinish>$on_finish</mis:actionOnFinish>
            <mis:headingMode>UsePointSetting</mis:headingMode>
            <mis:gimbalPitchMode>UsePointSetting</mis:gimbalPitchMode>
            <mis:powerSaveMode>false</mis:powerSaveMode>
            <mis:waypointType>LineStop</mis:waypointType>
            <mis:droneInfo>
              <mis:droneType>COMMON</mis:droneType>
              <mis:advanceSettings>false</mis:advanceSettings>
              <mis:droneCameras/>
              <mis:droneHeight>
                <mis:useAbsolute>false</mis:useAbsolute>
                <mis:hasTakeoffHeight>false</mis:hasTakeoffHeight>
                <mis:takeoffHeight>0.0</mis:takeoffHeight>
              </mis:droneHeight>
            </mis:droneInfo>
          </ExtendedData>
          <styleUrl>#waylineGreenPoly</styleUrl>
          <LineString>
            <tessellate>1</tessellate>
            <altitudeMode>relativeToGround</altitudeMode>
            <coordinates>$all_coordinates</coordinates>
          </LineString>
        </Placemark>
      </Document>
    </kml>""")

    for row in plan:
        name = row["label"]
        lon = row["lon"]
        lat = row["lat"]
        if lon[0] == "_":
            lon = lon[1:]
        if lat[0] == "_":
            lon = lat[1:]
        gimbal = row["gimbal"] if "speed" in row.keys() else DEFAULT_GIMBAL
        heading = row["heading"] if "heading" in row.keys() else DEFAULT_HEADING
        height = (
            float(row["altWGS84"]) + 10.0
        ) if "altWGS84" in row.keys() else DEFAULT_HEIGHT
        speed = row["speed"] if "speed" in row.keys() else DEFAULT_SPEED
        if "turnmode" in row.keys():
            turnmode = row["turnmode"] 
        else:
            turnmode = DEFAULT_TURNMODE
        if "actions_sequence" in row.keys():
            actions_sequence = row["actions_sequence"] 
        else:
            actions_sequence = DEFAULT_ACTIONS_SEQUENCE

        if (float(speed) > 15) or (float(speed) <= 0):
            sys.exit("speed should be >0 or <=15 m/s for {}".format(name))

        if gimbal and "." not in gimbal:
            gimbal = gimbal+".0"

        if turnmode == "AUTO":
            turnmode = "Auto"
        elif turnmode == "C":
            turnmode = "Clockwise"
        elif turnmode == "CC":
            turnmode = "Counterclockwise"
        else:
            sys.exit("turnmode shoud be AUTO C or CC for {}".format(name))

        if not heading:
            kml_str += waypoint_start_no_heading.substitute(
                turnmode=turnmode,
                waypoint_number=waypoint_number,
                speed=speed,
            )
        else:
            kml_str += waypoint_start.substitute(
                turnmode=turnmode,
                waypoint_number=waypoint_number,
                speed=speed,
                heading=heading,
                gimbal=gimbal
            )

        # Actions decoding
        if actions_sequence:
            action_list = actions_sequence.split(".")
            for action in action_list:
                if action == "SHOOT":
                    kml_str += shoot_template.substitute()
                elif action == "REC":
                    kml_str += record_template.substitute()
                elif action == "STOPREC":
                    kml_str += stoprecord_template.substitute()
                # Gimbal orientation
                elif action[0] == "G":
                    kml_str += gimbal_template.substitute(
                        gimbal_angle=action[1:])
                # Aircraft orientation
                elif action[0] == "A":
                    kml_str += aircraftyaw_template.substitute(
                        aircraftyaw=action[1:])
                elif action[0] == "H":
                    if float(action[1:]) < 500:
                        sys.exit(
                            "Hover length is in ms and should be >500  for {}".format(name)
                        )
                    kml_str += hover_template.substitute(
                        length=action[1:]
                    )
        elif hover >= 500:
            kml_str += hover_template.substitute(
                length=str(hover)
            )

        kml_str += "\n" + \
            waypoint_end.substitute(lon=lon, lat=lat, height=height,)+"\n"

        all_coordinates += all_coordinates_template.substitute(
            lon=lon, lat=lat, height=height)+" "
        waypoint_number += 1
# remove last space from coordinates string
    all_coordinates = all_coordinates[:-1]
    kml_str += xml_end.substitute(
        all_coordinates=all_coordinates,
        on_finish=on_finish
    )
    return kml_str

def csv2djipilot(data_path: str, kml_path: str, on_finish: str = "hover"):
    """csv2djipilot(data_path, kml_path, on_finish)

    Convert an existing CSV path with flight data into a KML path.

    @param  data_path (str) Input path.
    @param  kml_path (str)  Output path.
    @param  on_finish (str) Action to take on completion; defaults to "hover".
    """
    flight_plan_csv = _read_csv_file(data_path)
    kml_str = plan_2_kml(flight_plan_csv, on_finish)
    _write_file(path=kml_path, data=kml_str)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csvfile",
        type=argparse.FileType("r"),
        help="Specify csv input file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=DEFAULT_KML_PATH,
        help="Specify output file (default:stdout)"
    )
    parser.add_argument(
        "--onfinish",
        default="hover",
        choices=["hover", "gohome"],
        help="Aircraft action when finish. hover or gohome (default: %(default)s)."
    )
    args = parser.parse_args()

    if args.onfinish == "hover":
        on_finish = "Hover"
    elif args.onfinish == "gohome":
        on_finish = "GoHome"
    else:
        sys.exit("onfinish shoud be hover or gohome")

    CsvFile = args.csvfile.name

    data_path = args.csvfile.name
    kml_path = args.output.name

    csv2djipilot(data_path, kml_path, on_finish=on_finish)
