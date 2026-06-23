import laspy
import json
import os
import rasterio
import time

import  numpy as np

from pyproj import CRS
from rasterio.windows import from_bounds


class GeoBase:
    def __init__(self, filepath):
        self.filepath = filepath

"""GeoJSON
Handle GeoJSON file conversions here.
"""
class GeoJSON(GeoBase):
    def __init__(self, filepath):
        super().__init__(filepath)

        self.features = None
        self._load_features()


    def _load_features(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as gfp:
                data = json.load(gfp)
            self.features = data.get("features")


    def _generate_lines_from_coords(self, coords: list[(tuple, tuple)]):
        label = len(self.features)
        for coord in coords:
            self.features.append(
                    {
                        "type": "Feature",
                        "id": label,
                        "properties": {},  # MWM: this doesn't read properties?
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                coord[0],
                                coord[1]
                            ]
                        }
                    }
                )
            label += 1


    def _generate_points_from_coords(self, coords: list[tuple]):
        label = len(self.features)
        for coord in coords:
            self.features.append(
                    {
                        "type": "Feature",
                        "id": label,
                        "properties": {},
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                coord[0],
                                coord[1]
                            ]
                        }
                    }
                )
            label += 1


    def add_feature(self, coords: list[tuple], line: bool = False):
        if line:
            self._generate_lines_from_coords(coords)
        else:
            self._generate_points_from_coords(coords)


    def _get_geojson(self):
        return {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::32610"
                    }
                },
                "features": self.features
            }


    def write(self):
        geodata = self._get_geojson()
        with open(self.filepath) as gfp:
            json.dump(geodata, gfp)


    def __str__(self):
        return str(self._get_geojson())


class GeoPointCloud(GeoBase):

    def __init__(self, filepath):
        super().__init__(filepath)
        self._load_pointcloud()


    def _load_pointcloud(self):
        try:
            self.las = laspy.read(self.filepath)
        except laspy.LazBackendException as e:
            raise RuntimeError("Cannot read .laz file. Install lazrs: pip install laspy[lazrs]") from e
        self.points = np.vstack((self.las.x, self.las.y, self.las.z)).T
        self.intensity = self.las.intensity
        self.classification = self.las.classification
        self.crs = CRS.from_wkt(self.las.header.parse_crs().to_wkt())


    def summary(self):
        return {
            "File": self.filepath,
            "Number of Points": len(self.points),
            "CRS": self.crs.to_string(),
            "Bounds": {
                "X": (float(np.min(self.points[:, 0])), float(np.max(self.points[:, 0]))),
                "Y": (float(np.min(self.points[:, 1])), float(np.max(self.points[:, 1]))),
                "Z": (float(np.min(self.points[:, 2])), float(np.max(self.points[:, 2])))
            }
        }


    def get_points(self):
        return self.points


    def sample_surface_height(self, xmin, ymin, xmax, ymax):
        """Return average height (z) of points within given XY bounding box."""
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if np.any(mask):
            return float(np.mean(z[mask]))
        else:
            return None  # or np.nan


    def filter_by_classification(self, class_id):
        mask = self.classification == class_id
        return self.points[mask]


class GeoTIFF(GeoBase):

    def __init__(self, filepath):
        super().__init__(filepath)
        self._load_tiff()


    def _load_tiff(self):
        with rasterio.open(self.filepath) as src:
            self.data = src.read()  # All bands
            self.meta = src.meta
            self.bounds = src.bounds
            self.crs = src.crs
            self.transform = src.transform
            self.width = src.width
            self.height = src.height
            self.count = src.count


    def sample_surface_height(self, xmin, ymin, xmax, ymax):
        """Return average height (z) from raster within given XY bounding box."""
        window = from_bounds(xmin, ymin, xmax, ymax, transform=self.transform)

        # Convert to integer pixel window
        row_start, row_stop = int(window.row_off), int(window.row_off + window.height)
        col_start, col_stop = int(window.col_off), int(window.col_off + window.width)

        subset = self.data[0, row_start:row_stop, col_start:col_stop]  # band 1
        if subset.size == 0:
            return None

        nodata = self.meta.get("nodata")
        if nodata is not None:  # if there is data
            subset = np.ma.masked_equal(subset, nodata)  # nodata entry search
            if subset.mask.all():  # redundant all nodata check
                return None

        return float(subset.mean())


    def summary(self):
        return {
            "File": self.filepath,
            "CRS": self.crs,
            "Bounds": {
                "X": (self.bounds.left, self.bounds.right),
                "Y": (self.bounds.bottom, self.bounds.top)
            },
            "Size": (self.width, self.height),
            "Bands": self.count
        }
