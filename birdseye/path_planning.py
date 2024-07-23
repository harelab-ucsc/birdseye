"""path_planning.py - KML interface for path planning on DJI drones.

Description:
    Use this library to load interesting stuff to DJI drone controllers.
"""
import csv
import logging
from typing import Union


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

def write_plan(data: list[dict]):
    """
    """
    logging.info("WIP")
