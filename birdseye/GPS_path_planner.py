"""GPS_path_planner - convert clicks into a flight plan.
Description:
    Translate input clicks (CSV) into a complete flight plan (KML). The output
    file can then be loaded to an SD card and used by a compatible drone.

Usage:
    TODO

Todo:
    TODO
"""
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import DBSCAN
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv
import os
import sys
import utm
import simplekml

from path_planning import write_plan
from TSP import tsp


EPS = 1.0
MIN_SAMPLES = 2


def _write_csv_file(path: str, rows: list[str]):
    """Write rows of CSV to a file.
    """
    with open(path, "w+") as csvp:
        writer = csv.writer(csvp, delimiter=",")
        writer.writerows(rows)

def generate_csv_from_plan(plan, fpath):
    """Write the details of a flight plan to a CSV file.
    """
    lines = []
    lines.append(["lat","lon","point_name"])
    [lines.append([
        str(pos[0]),
        str(pos[1]),
        str(index)
    ]) for index, pos in enumerate(plan)]
    _write_csv_file(fpath, lines)


def build_polygon(
        pt: tuple[float],
        r: float,
        n: int
    ) -> tuple[tuple[float]]:
    """build_polygon

    Description:
        Translate a point, radius, and number of sides into a set of points
        defining a polygon.

    Args:
        pt (:obj:`tuple` of :obj:`float`): [x,y]
        r (float): "Radius" == absolute distance from `pt` of each vertex.
        n (int): Number of sides/vertices for polygon.
    """
    # Generate polygons around each point provided.
    thetas = lambda n: [((i * 2 * np.pi) / n) for i in range(n)]
    return [
        [pt[0] + r * np.cos(th),
         pt[1] + r * np.sin(th)] for th in thetas(n)
    ]

def build_dji_plan(
        do_tsp: bool,
        do_whifferdill: bool,
        fp_in: str,
        fp_out: str,
        w_rad: float,
        w_sides: int
    ):
    """
    Args:
        do_tsp          (bool)  :
        do_whifferdill  (bool)  :
        fp_in           (str)   :
        fp_out          (str)   :
        w_rad           (float) :
        w_sides         (int)   :

    Todo:
        * Integrate args into the below.
        * Cleanup.
    """
    # NOTE: Set data path here.
    #fp_in = "catch/data.csv"
    #fp_in = "Documents/hare/birdseye/birdseye/catch/data.csv"
    #savename = "parsed_flight/plan.kml"
    #savename = fp_out
    #plan_kml = os.path.join(os.path.expanduser('~'), savename)
    clicks_csv = fp_in
    plan_kml = fp_out
    

    LLrep = []
    UTMrep = []
    dists = []
    with open(clicks_csv) as clicks:
        reader = csv.reader(clicks)
        for line in reader:
            # break down line
            try:
                u = utm.from_latlon(float(line[0]), float(line[1]))
                ll = '(' + ','.join(line[:2]) + ')'
                print('lat/lon click location: ', ll)
                print('    utm conversion: ', u)
                tag = int(line[-1][-1])
                UTMrep.append([u[0], u[1]])
                # for _ in range(5):
                #     UTMrep.append(mvn.rvs(mean=[u[0], u[1]], cov=0.5).tolist())  # clicks in UTM coordinates, meter base unit
            except:
                continue
    UTMrep = np.array(UTMrep)
    print()

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(UTMrep)  # cluster in the UTM/cartesian representation
    labels = dbscan.labels_
    print('labels: ', labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("    Estimated number of clusters: %d" % n_clusters_)
    print("    Estimated number of noise points: %d" % n_noise_)
    print()

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    # print('core_samples_mask: ', core_samples_mask)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    waypoints = []
    for k, col in zip(unique_labels, colors):
        tmp = []
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            skip_convhull = True
        else:
            skip_convhull = False

        class_member_mask = labels == k

        xy = UTMrep[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )
        for pt in xy:
            if skip_convhull:
                waypoints += build_polygon(pt, w_rad, w_sides)
            else:
                tmp += build_polygon(pt, w_rad, w_sides)

        xy = UTMrep[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
        for pt in xy:
            if skip_convhull:
                waypoints += build_polygon(pt, w_rad, w_sides)
            else:
                tmp += build_polygon(pt, w_rad, w_sides)

        tmp = np.array(tmp)

        if len(tmp) != 0:
            hull = ConvexHull(tmp)
            pts = tmp[hull.vertices]
            plt.plot(pts[:,0], pts[:,1], 'o', markerfacecolor='g', markeredgecolor="k", markersize=10)
            waypoints += list(pts)

    waypoints = np.array(waypoints)
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    for i, pti in enumerate(waypoints):
        tmp = []
        for j, ptj in enumerate(waypoints[i+1:]):
            tmp.append(np.linalg.norm(pti-ptj))
        dists.append(tmp)

    plan = []
    places = [i for i in range(len(waypoints))]
    out = tsp(places, dists) if do_tsp else places
    for pt in waypoints[out]:
        plan.append(list(utm.to_latlon(pt[0], pt[1], u[-2], u[-1])))
    plan = np.array(plan)

    generate_csv_from_plan(plan, fp_out)

    spt = out[0]
    plt.plot(waypoints[spt,0], waypoints[spt,1], 'o', markerfacecolor='r', markeredgecolor='k', markersize=10)
    for pt in out[1:]:
        plt.plot([waypoints[spt,0], waypoints[pt,0]], [waypoints[spt,1], waypoints[pt,1]], 'k')
        plt.plot(waypoints[pt,0], waypoints[pt,1], 'o', markerfacecolor='b', markeredgecolor='b', markersize=4)
        spt = pt
    plt.show()
    """
    kml=simplekml.Kml()
    for i in out:
        print(plan[i])
        kml.newpoint(name=str(places[i]), coords=[(plan[i][1], plan[i][0])])
    kml.save(plan_kml)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="catch/data.csv",
        help="Specify path to input [CSV] file.",
        type=str
    )
    parser.add_argument(
        "-o",
        "--output",
        default="flights/pilot.kml",
        help="Specify path to output [KML] file; defaults to STDOUT.",
        type=str
    )
    parser.add_argument(
        "-r",
        "--radius",
        default=1.0,
        help="Specify the radius of created Whifferdill [if enabled].",
        type=float
    )
    parser.add_argument(
        "-s",
        "--sides",
        default=4,
        help="Specify the number of sides of created Whifferdill [if enabled].",
        type=int
    )
    parser.add_argument(
        "-t",
        "--tsp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Find the shortest path between each point visited?"
    )
    parser.add_argument(
        "-w",
        "--whifferdill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Whifferdill pattern around each point?"
    )
    args = parser.parse_args()

    build_dji_plan(
        do_tsp=args.tsp,
        do_whifferdill=args.whifferdill,
        fp_in=args.input,
        fp_out=args.output,
        w_rad=args.radius,
        w_sides=args.sides
    )
