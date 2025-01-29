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
import copy as cp
import csv
import os
import sys
import utm
#import simplekml

# from TSP import tsp
import fast_tsp

from kml import plan_2_kml

EPS = 2
MIN_SAMPLES = 3

DJI_MAX_POINTS = 95
DJI_MIN_DISTANCE = 3500  # millimeters -> 3.5m...
# there is a more formal post (from DroneDeploy) which claims that the min is
# 5.0m, but we have done flights which contradict that figure (4pts @ 1m
# whifferdill ).

SAVE = True


def _write_csv_file(path: str, rows: list[str]):
    """Write rows of CSV to a file.
    """
    with open(path, "w+") as csvp:
        writer = csv.writer(csvp, delimiter=",")
        writer.writerows(rows)

def _write_file(path: str, data: str):
    """Write a string to a file.
    """
    with open(path, "w") as fp:
        writer = fp.write(data)

def generate_csv_from_plan(plan, d_hover, fpath):
    """Write the details of a flight plan to a CSV file.
    """
    lines = []
    lines.append(["lat","lon","point_name", "actions_sequence"])
    [lines.append([
        str(pos[0]),
        str(pos[1]),
        str(index),
        f"H{d_hover}"
    ]) for index, pos in enumerate(plan)]
    _write_csv_file(fpath, lines)

def _format_plan(
    plan: list[list],
    d_hover: int = 0.0
) -> list[dict]:
    """_format_plan(plan, d_hover) -> plan_out

    Reformat the plan as a list of waypoints with path parameter customization.

    @param  plan (list[list])   Plan as a list of lat/lon.
    @param  d_hover (int)       Hover duration (seconds).
    """
    plan_out = []
    template = {
        "lat": "",
        "lon": "",
        "label": "",
        "actions_sequence": ""
    }
    for index, pos in enumerate(plan):
        row = cp.deepcopy(template)
        row["lat"] = str(pos[0])
        row["lon"] = str(pos[1])
        row["label"] = str(index)
        row["actions_sequence"] = f"H{d_hover}"
        plan_out.append(row)
    return plan_out

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
    if r == 0.0:
        return [list(pt)]
    else:
        thetas = lambda n: [((i * 2 * np.pi) / n) for i in range(n)]
        return [
            [pt[0] + r * np.cos(th),
             pt[1] + r * np.sin(th)] for th in thetas(n)
        ]


def preprocessWaypoints(waypoints, min_gap=DJI_MIN_DISTANCE):
    wpts = np.array(waypoints)
    dists = metrics.pairwise_distances(wpts)
    dists *= 1000  # convert to mm from m
    dists = dists.astype(np.int32)
    # plt.imshow(dists)
    # plt.show()

    while dists[np.nonzero(dists)].min() < min_gap:
        d_tmp = np.nonzero(dists)
        indx = np.where(dists == dists[d_tmp].min())
        u, v = indx[0][0], indx[1][0]
        if u > v:
            a = waypoints.pop(u)
            b = waypoints.pop(v)
        else:
            a = waypoints.pop(v)
            b = waypoints.pop(u)
        new = [(a[0]+b[0])/2, (a[1]+b[1])/2]
        waypoints.append(new)
        wpts = np.array(waypoints)
        dists = metrics.pairwise_distances(wpts)
        dists *= 1000  # convert to mm from m
        dists = dists.astype(np.int32)
    # plt.imshow(dists)
    # plt.show()
    print(dists[np.nonzero(dists)].min())
    # print(np.where(dists == 0))
    return dists, wpts


def build_dji_plan(
        d_hover: float,
        do_tsp: bool,
        do_whifferdill: bool,
        fp_in: str,
        fp_out: str,
        w_rad: float,
        w_sides: int
    ):
    """
    Args:
        d_hover         (float) :
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
                # print('lat/lon click location: ', ll)
                # print('    utm conversion: ', u)
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
    # print('labels: ', labels)
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
    plt.rcParams['figure.figsize'] = [15,15]
    for k, col in zip(unique_labels, colors):
        tmp = []
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            skip_convhull = True
        # elif w_rad == 0.0:
        #     skip_convhull = True
        else:
            skip_convhull = False

        class_member_mask = labels == k

        xy = UTMrep[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            # markerfacecolor=tuple(col),
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=5,
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
            # markerfacecolor=tuple(col),
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=5,
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
            # plt.plot(pts[:,0], pts[:,1], 'o', markerfacecolor='g', markeredgecolor="k", markersize=10)
            waypoints += list(pts)

    dists, waypoints = preprocessWaypoints(waypoints)

    if do_tsp:
        out = fast_tsp.find_tour(dists)
    else:
        out = range(len(waypoints))
    spt = out[0]
    for pt in out[1:]:
        plt.plot([waypoints[spt,0], waypoints[pt,0]], [waypoints[spt,1], waypoints[pt,1]], 'k')
        plt.plot(waypoints[pt,0], waypoints[pt,1], 'o', markerfacecolor='xkcd:neon green', markeredgecolor='k', markersize=5)
        spt = pt
    plt.plot([waypoints[spt,0], waypoints[out[0],0]], [waypoints[spt,1], waypoints[out[0],1]], 'k')

    plan = []
    for pt in waypoints[out]:
        plan.append(list(utm.to_latlon(pt[0], pt[1], u[-2], u[-1])))
    # plan = np.array(plan)
    print(len(plan)//DJI_MAX_POINTS, len(plan)%DJI_MAX_POINTS)

    print(f"Formatting flight plan...")
    plan_formatted = _format_plan(plan, int(d_hover * 1000))
    plan_kml = plan_2_kml(plan_formatted)
    print("DONE.")
    print(f"Generating flight plan at {fp_out}...")
    _write_file(fp_out, plan_kml)
    """
    generate_csv_from_plan(
        plan,
        int(d_hover * 1000),
        fp_out
    )
    """
    print("DONE.")

    plt.tick_params(axis='x', which='both', bottom=False,
                top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False,
                left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    if SAVE:
        fig, ax = plt.subplots(figsize=(15,15))
        ax.plot(UTMrep[:,0], UTMrep[:,1],
            "o",
            # markerfacecolor=tuple(col),
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=10,
            )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.savefig('clicks.png', transparent=True)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--duration_hover",
        default=3.0,
        help="Specify duration of hover at each point.",
        type=float
    )
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
        default="flights/plan.kml",
        help="Specify path to output [KML] file; defaults to 'plan.kml'.",
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
        d_hover=args.duration_hover,
        do_tsp=args.tsp,
        do_whifferdill=args.whifferdill,
        fp_in=args.input,
        fp_out=args.output,
        w_rad=args.radius,
        w_sides=args.sides
    )
