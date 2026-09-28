"""
@file       clicks2geojson.py

Convert CSV output from the Clicker to GeoJSON format for use with QGIS and
OASIS Sim.

@author     HARE Lab
@author     nubby   (jlee211@ucsc.edu)

@date       8 Jul 2025
@version    1.0.0
"""
import csv
import json
import utm


"""_read_csv(path)
@param
@return

@todo   If there is a header row, read as a dictionary.
"""
def _read_csv(path: str) -> list:
    fieldnames = (
            "latitude",
            "longitude",
            "altitude_wgs",
            "altitude_wsl",
            "gps_status",
            "label"
            )
    data = []
    with open(path, "r") as cvp:
        csv_reader = csv.DictReader(cvp, fieldnames=fieldnames)
        for row in csv_reader:
            data.append(row)
    return data

"""_write_json(data, path)
@param  data
@param  path
"""
def _write_json(data: dict, path: str):
    with open(path, "w") as gfp:
        json.dump(data, gfp)

"""_generate_point_from_coord(poi_coord, label)

Convert coordinates into a GeoJSON point format.

@param
@return
"""
def _generate_point_from_coord(poi_coord: tuple, label: int) -> dict:
    return {
            "type": "Feature",
            "id": label,
            "properties": {},
            "geometry": {
                "type": "Point",
                "coordinates": [poi_coord[0], poi_coord[1]]
                }
            }

"""_generate_geojson_from_coords(coords, name)

Convert a list of coordinates to a GeoJSON format.

@param
@param
@return
"""
def _generate_geojson_from_coords(
        crs: str,
        coords: list[tuple],
        name: str) -> dict:
    crs_lut = {
            "default": "OGC::CRS84",
            "utm10n": "EPSG::32610",    # Santa Cruz, CA, USA.
            "wgs84": "OGC::CRS84"       # WGS84 lat/lon.
            }
    label = 0
    features = []
    for coord in coords:
        features.append(_generate_point_from_coord(coord, label))
        label += 1
    return {
            "type": "FeatureCollection",
            "name": name,
            "crs": {
                "type": "name",
                "properties": {
                    "name": f"urn:ogc:def:crs:{crs_lut[crs]}"
                    }
                },
            "features": features
            }

"""convert_clicks2geojson(clicks)

Convert raw clicks recorded by a Clicker into GeoJSON format.

@param  clicks
@param  use_utm Convert to UTM?
@return 

@todo   When the clicker can report UTM on its own, make that ingestible.
"""
def convert_clicks2geojson(
        clicks: list[dict],
        name: str = "Clicks",
        use_utm: bool = False) -> dict:

    # Extract lat/lon for preprocessing.
    lat_coords = [click["latitude"] for click in clicks]
    lon_coords = [click["longitude"] for click in clicks]
    crs = ""

    # Convert to UTM if desired, else bundle coordinates as lat/lon. 
    if use_utm:
        coords = [utm.from_latlon(
            lat_coord, lon_coord
            ) for lat_coord, lon_coord in zip(lat_coords, lon_coords)]
        try:
            crs = "".join([str(coords[0][-2]), coords[0][-1]])
        except Exception as e:
            print(e)
    else:
        coords = zip(lat_coords, lon_coords)
        crs = "wgs84"

    # Convert named clicks with CRS to GeoJSON format.
    return _generate_geojson_from_coords(crs, coords, name)

"""write_clicks2geojson(path_clicks, path_geojson)

Convert raw clicks recorded by a Clicker into GeoJSON format and write to a
file.

@param
@param
"""
def write_clicks2geojson(path_clicks: str, path_geojson: str):
    data_csv = _read_csv(path_clicks)
    geojson_out = convert_clicks2geojson(data_csv)
    _write_json(geojson_out, path_geojson)


# Testing.
"""_test_clicks2geojson()

Test harness for the clicks2geojson module.

@todo   Expand coverage.
"""
def _test_clicks2geojson():
    sample_wgs84_data = [ {
        "latitude": 36.957522921,
        "longitude": -122.058309900,
        "altitude_wgs": -2.258000,
        "altitude_wsl": 28.402000,
        "gps_status": 2,
        "label": "label 3"
    }, {
        "latitude": 36.957522922,
        "longitude": -122.058309901,
        "altitude_wgs": -2.258001,
        "altitude_wsl": 28.402001,
        "gps_status": 2,
        "label": "label 3"
        } ]
    geojson_out = convert_clicks2geojson(sample_wgs84_data)
    print(str(geojson_out))

    # TODO: Include these as test files/output locations.
    path_clicks = "../20250224_data.csv"
    path_geojson = "test.geojson"
    write_clicks2geojson(path_clicks, path_geojson)
    

# TODO: Move to a separate test file.
if __name__ == "__main__":
    _test_clicks2geojson()
