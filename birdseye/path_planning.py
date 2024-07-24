"""path_planning.py - KML interface for path planning on DJI drones.

Description:
    Use this library to load interesting stuff to DJI drone controllers.
"""
import csv
import logging
from typing import Union


DEFAULT_ACTIONS_SEQUENCE: Union[str, None] = None
DEFAULT_GIMBAL: Union[float, None] = None
DEFAULT_HEADING: Union[float, None] = None
DEFAULT_HEIGHT = 10    # m
DEFAULT_SPEED = 2.3    # m/s
DEFAULT_TURNMODE = 'AUTO'


# Helpers.
def _write_file(path: str, data: any):
    with open(path, "w+") as fp:
        fp.write(data)

# WIP.
class FlightPath(object):
    """FlightPath - Format paths for drones.

    Args:
        name (Optional[str])    : Flight path nickname.
    """
    def __init__(self, name=None):
        self.name = name
        self.flight_plan = []

    def load_flight_plan(self, fpath: Union[str,list[dict]]):
        """Either load a flight plan directly to this 

        Description:
            Flight path data must be structured/-able in the format:
                {"point_name": str, "lat": float, "lon": float}

        Args:
            flight_plan (Union[str,list[dict]]) :   Either the path to a CSV
                                                    file or a list of dicts.
        """
        if not isinstance(flight_plan, list[dict]):
            try:
                self.import_csv(flight_plan)
            except:
                logging.error(
                    f"Could not import flight plan from {flight_plan}."
                )
        else:
            self.flight_plan = flight_plan


    def import_csv(self, cpath: str):
        """
        Args:
            cpath (str):    File path.
        """
        with open(cpath) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                self.flight_plan.append(line)

def write_plan(data: list[dict], kmlpath: str):
    """
    """
    logging.info("Converting flight plan to DJI Pilot KML format...")
    for row in data:
        name = row['point_name']
        lon = row['lon']
        lat = row['lat']
        # NOTE: start legacy code.
        if lon[0] == '_':
            lon = lon[1:]
        if lat[0] == '_':
            lon = lat[1:]
        # NOTE: end legacy code.

        gimbal = row['gimbal'] if 'speed' in row.keys() else DEFAULT_GIMBAL
        heading = row['heading'] if 'heading' in row.keys() else DEFAULT_HEADING
        height = row['height'] if 'height' in row.keys() else DEFAULT_HEIGHT
        speed = row['speed'] if 'speed' in row.keys() else DEFAULT_SPEED



def csv2djipilot():
    XML_string = """<?xml version="1.0" encoding="UTF-8"?>

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
            <mis:actionOnFinish>$ON_FINISH</mis:actionOnFinish>
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

    with open(CsvFile, newline='') as csvfile:
        # TODO(nubby): allow for the import of other delimiters.
        # NOTE - Required attributes:
        #           * point_name
        #           * lat
        #           * lon
        csv_lines = csv.DictReader(csvfile)
        for row in csv_lines:
            name = row['point_name']
            lon = row['lon']
            lat = row['lat']
            if lon[0] == '_':
                lon = lon[1:]
            if lat[0] == '_':
                lon = lat[1:]
            gimbal = row['gimbal'] if 'speed' in row.keys() else DEFAULT_GIMBAL
            heading = row['heading'] if 'heading' in row.keys() else DEFAULT_HEADING
            height = row['height'] if 'height' in row.keys() else DEFAULT_HEIGHT
            speed = row['speed'] if 'speed' in row.keys() else DEFAULT_SPEED
            if 'turnmode' in row.keys():
                turnmode = row['turnmode'] 
            else:
                turnmode = DEFAULT_TURNMODE
            if 'actions_sequence' in row.keys():
                actions_sequence = row['actions_sequence'] 
            else:
                actions_sequence = DEFAULT_ACTIONS_SEQUENCE

            if (float(speed) > 15) or (float(speed) <= 0):
                sys.exit('speed should be >0 or <=15 m/s for {}'.format(name))
            """
            if '.' not in speed:
                speed = speed+'.0'
            """

            if gimbal and '.' not in gimbal:
                gimbal = gimbal+'.0'

            if turnmode == 'AUTO':
                turnmode = 'Auto'
            elif turnmode == 'C':
                turnmode = 'Clockwise'
            elif turnmode == 'CC':
                turnmode = 'Counterclockwise'
            else:
                sys.exit('turnmode shoud be AUTO C or CC for {}'.format(name))

            if not heading:
                XML_string += waypoint_start_no_heading.substitute(
                    turnmode=turnmode,
                    waypoint_number=waypoint_number,
                    speed=speed,
                )
            else:
                XML_string += waypoint_start.substitute(
                    turnmode=turnmode,
                    waypoint_number=waypoint_number,
                    speed=speed,
                    heading=heading,
                    gimbal=gimbal
                )

            # Actions decoding
            if actions_sequence:
                action_list = actions_sequence.split('.')
                for action in action_list:
                    if action == 'SHOOT':
                        XML_string += shoot_template.substitute()
                    elif action == 'REC':
                        XML_string += record_template.substitute()
                    elif action == 'STOPREC':
                        XML_string += stoprecord_template.substitute()
                    # Gimbal orientation
                    elif action[0] == 'G':
                        XML_string += gimbal_template.substitute(
                            gimbal_angle=action[1:])
                    # Aircraft orientation
                    elif action[0] == 'A':
                        XML_string += aircraftyaw_template.substitute(
                            aircraftyaw=action[1:])
                    elif action[0] == 'H':
                        if float(action[1:]) < 500:
                            print(float(action[1:]))
                            sys.exit(
                                'Hover length is in ms and should be >500  for {}'.format(name))
                        XML_string += hover_template.substitute(
                            length=action[1:])

            XML_string += "\n" + \
                waypoint_end.substitute(lon=lon, lat=lat, height=height,)+"\n"

            all_coordinates += all_coordinates_template.substitute(
                lon=lon, lat=lat, height=height)+" "
        waypoint_number += 1
# remove last space from coordinates string
    all_coordinates = all_coordinates[:-1]
    XML_string += xml_end.substitute(all_coordinates=all_coordinates,
                                     ON_FINISH=ON_FINISH)
    #_write_file(path=args.output, data=XML_string)
    with args.output as outpoofile:
        outpoofile.write(XML_string)
    

