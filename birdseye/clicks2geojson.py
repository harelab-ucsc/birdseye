"""
@file       clicks2geojson.py

Convert CSV output from the Clicker to GeoJSON format for use with QGIS and
OASIS Sim.

@author     HARE Lab
@author     nubby   (jlee211@ucsc.edu)

@date       8 Jul 2025
@version    0.0.9
"""

"""_generate_point_from_poi(poi_coord)

Convert UTM coordinates into a GeoJSON point format.

@param
@return
"""
def _generate_point_from_poi(poi_coord: tuple, label: int) -> dict:
    return {
            "type": "Feature",
            "id": label,
            "properties": {},
            "geometry": {
                "type": "Point",
                "coordinates": [poi_coord[0], poi_coord[1]]
                }
            }

"""_generate_points_from_pois(gcp_coords, resolution)

Bound the polygon defined by GCP coordinates within a rectangle and return
GeoJSON-formatted dictionaries that form a "brownie pan" grid.

@param  gcp_coords  [easting, northing, zone_number, zone_letter]
@param  resolution  In meters; defines the length of one side of square nodes.
@return List of GeoJSON-formatted dictionaries defining the grid.
"""
def _generate_points_from_pois(poi_coords: list[tuple]) -> dict:
    # TODO(nubby)
    name = ""
    crs = ""
    label = 0
    features = []
    for coord in poi_coords:
        features.append(_generate_point_from_poi(coord, label))
        label += 1
    return {
            "type": "FeatureCollection",
            "name": name,
            "crs": {
                "type": "name",
                "properties": {
                    "name": crs
                    }
                },
            "features": features
            }

def _test_clicks2geojson():
    pass

# TODO: Move to a separate test file.
if __name__ == "__main__":
    _test_clicks2geojson()
