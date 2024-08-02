#!/usr/bin/env python3
"""
Todo:
    * Set header if not supplied.
"""
from string import Template
from typing import Union
import csv
import sys
import argparse


#_CURRENT_ALTITUDE = 30
DEFAULT_ACTIONS_SEQUENCE: Union[str, None] = None
DEFAULT_GIMBAL: Union[float, None] = None
DEFAULT_HEADING: Union[float, None] = None
DEFAULT_HEIGHT = 10    # m
DEFAULT_NAME = "xatu"
DEFAULT_SPEED = 2.3    # m/s
DEFAULT_TURNMODE = 'AUTO'

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', type=argparse.FileType('r'),
                    help="Specify csv input file")
#parser.add_argument('-outputfile',type=string, required=False, default="pilot.kml")
parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                    default=sys.stdout, help="Specify output file (default:stdout)")
parser.add_argument(
    '--onfinish', default='hover',
    choices=['hover', 'gohome'],
    help='Aircraft action when finish. hover or gohome (default: %(default)s)'
)
args = parser.parse_args()

if args.onfinish == 'hover':
    ON_FINISH = "Hover"
elif args.onfinish == 'gohome':
    ON_FINISH = "GoHome"
else:
    sys.exit('onfinish shoud be hover or gohome')

CsvFile = args.csvfile.name

print(f'{CsvFile} to {args.output.name}')
#CsvFile = 'exemple.csv'
#CsvFile = 'exemple_simple.csv'
CSV_HEADER = False

def _write_file(path: str, data: any):
    with open(path, "w+") as fp:
        fp.write(data)

"""
Args:
    author      (str)
    now         (str((int))):           Epoch file creation time in ms.
    height      (str(float))
    WAYPOINTS   (str(list(Template]))): List of waypoint templates.
"""
template_plan_base = Template("""
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="http://www.dji.com/wpmz/1.0.2">
<Document>

    <!-- Step 1: Implement File Creation Information. -->
    <wpml:author>$author</wpml:author>
    <wpml:createTime>$now</wpml:createTime>
    <wpml:updateTime>$now</wpml:updateTime>

    <!-- Step 2: Setup Mission Configuration. -->
    <wpml:missionConfig>
        <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
        <wpml:finishAction>goHome</wpml:finishAction>
        <wpml:exitOnRCLost>goContinue</wpml:exitOnRCLost>
        <wpml:executeRCLostAction>hover</wpml:executeRCLostAction>
        <wpml:takeOffSecurityHeight>$height</wpml:takeOffSecurityHeight>
        <wpml:globalTransitionalSpeed>$speed</wpml:globalTransitionalSpeed>
        <wpml:droneInfo>
          <!-- Declare drone model with M300. -->
            <wpml:droneEnumValue>60</wpml:droneEnumValue>
        </wpml:droneInfo>
        <!-- Try to get rid of this. -->
        <wpml:payloadInfo>
            <!-- Declare payload model with Matrice 3D camera. -->
            <wpml:payloadEnumValue>80</wpml:payloadEnumValue>
            <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
        </wpml:payloadInfo>
    </wpml:missionConfig>
    $WAYPOINTS
</Document>
</kml>
""")

"""
Args:
    waypoint_id (str(int)):     Index of waypoint. 
    height_mode (str):          RelativeToStartPoint is good.
    height      (str(float))
    hover_time
    speed
"""
template_waypoint = Template("""
<!-- Step 3: Setup A Folder for Waypoint Template -->
<Folder>
    <wpml:templateType>waypoint</wpml:templateType>
    <wpml:useGlobalTransitionalSpeed>0</wpml:useGlobalTransitionalSpeed>
    <wpml:templateId>$waypoint_id</wpml:templateId>
    <wpml:waylineCoordinateSysParam>
        <wpml:heightMode>$height_mode</wpml:heightMode>
        <!-- Remove?    -->
        <wpml:globalShootHeight>$height</wpml:globalShootHeight>
        <wpml:positioningType>GPS</wpml:positioningType>
        <!-- Remove?    -->
        <wpml:surfaceFollowModeEnable>1</wpml:surfaceFollowModeEnable>
        <!-- Remove?    -->
        <wpml:surfaceRelativeHeight>$height</wpml:surfaceRelativeHeight>
    </wpml:waylineCoordinateSysParam>
    <wpml:autoFlightSpeed>$speed</wpml:autoFlightSpeed>
    <!-- Remove?    -->
    <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
    <wpml:globalWaypointHeadingParam>
        <wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode>
        <!-- Verify.    -->
        <wpml:waypointHeadingAngle>0</wpml:waypointHeadingAngle>
        <!-- Verify.    -->
        <wpml:waypointPoiPoint>$lat,$lon,$height</wpml:waypointPoiPoint>
        <!-- Verify. [clockwise, counterClockwise, followBadAngle]    -->
        <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>
    </wpml:globalWaypointHeadingParam>
    <!-- Verify.    -->
    <wpml:globalWaypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:globalWaypointTurnMode>
    <!-- Verify. [0,1]   -->
    <wpml:globalUseStraightLine>0</wpml:globalUseStraightLine>
    <Placemark>
        <Point>
            <!-- Fill longitude and latitude here -->
            <coordinates>
                $lon,$lat
            </coordinates>
        </Point>
        <wpml:index>$waypoint_id</wpml:index>
        <wpml:ellipsoidHeight>$height</wpml:ellipsoidHeight>
        <wpml:height>$height</wpml:height>
        <wpml:useGlobalHeight>1</wpml:useGlobalHeight>
        <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
        <wpml:useGlobalHeadingParam>1</wpml:useGlobalHeadingParam>
        <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
        <wpml:gimbalPitchAngle>0</wpml:gimbalPitchAngle>
        <wpml:actionGroup>
            <wpml:actionGroupId>$waypoint_id</wpml:actionGroupId>
            <wpml:actionGroupStartIndex>1</wpml:actionGroupStartIndex>
            <wpml:actionGroupEndIndex>1</wpml:actionGroupEndIndex>
            <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
            <wpml:actionTrigger>
                <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>
            </wpml:actionTrigger>
            <!-- Declare the action: hover. -->
            <wpml:action>
                <wpml:actionId>$waypoint_id</wpml:actionId>
                <wpml:actionActuatorFunc>hover</wpml:actionActuatorFunc>
                <wpml:actionActuatorFuncParam>
                    <wpml:hoverTime>$hover_time</wpml:hoverTime>
                </wpml:actionActuatorFuncParam>
            </wpml:action>
        </wpml:actionGroup>
    </Placemark>
</Folder>
""")

def csv2djipilot():
    with open(CsvFile, newline='') as csvfile:
        # TODO(nubby): allow for the import of other delimiters.
        # NOTE - Required attributes:
        #           * point_name
        #           * lat
        #           * lon
        csv_lines = csv.DictReader(csvfile)
        waypoint_templates = []
        author = DEFAULT_NAME
        now = 1
        height = 20
        height_mode = "relativeToStartPoint"
        hover_time = 3
        speed = 2.3
        for waypoint_id, row in enumerate(csv_lines):
            print(row, waypoint_id)
            name = row['point_name']
            lon = row['lon']
            lat = row['lat']
            waypoint_templates.append(
                template_waypoint.substitute(
                    waypoint_id=str(waypoint_id),
                    height=str(height),
                    height_mode=height_mode,
                    hover_time=str(hover_time),
                    lat=str(lat),
                    lon=str(lon),
                    speed=str(speed)
                )
            )
        kml_str_out = template_plan_base.substitute(
            author=author,
            now=str(now),
            height=str(height),
            speed=str(speed),
            WAYPOINTS="".join(waypoint_templates)
        )
        print(kml_str_out)

    with args.output as outpoofile:
        outpoofile.write(kml_str_out)
    

if __name__ == "__main__":
    csv2djipilot()
