from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

import csv
import os
import utm
import simplekml

from TSP import tsp


EPS = 0.5
MIN_SAMPLES = 3


def _write_csv_file(path: str, rows: list[str]):
    """
    """
    with open(path, "w+") as csvp:
        writer = csv.writer(csvp, delimiter=",")
        writer.writerows(rows)

def generate_csv_from_plan(plan):
    """
    """
    header = ["lat,lon,point_name"]
    lines = ["{},{},{}".format(
        str(pos[0]),
        str(pos[1]),
        str(index)
    ) for index, pos in enumerate(plan)]
    _write_csv_file("balthazar.csv", [header + lines])


def build_dji_plan(generate: bool = True):
    """
    """
    filepath = "catch/data.csv"
    clicks_csv = os.path.join(os.path.expanduser('~'), filepath)
    savename = "parsed_flight/plan.kml"
    plan_kml = os.path.join(os.path.expanduser('~'), savename)

# thetas = [i*60*np.pi/180 for i in range(6)]
    thetas = [i*90*np.pi/180 for i in range(4)]
    hex = lambda pt: [[pt[0] + np.cos(th), pt[1] + np.sin(th)] for th in thetas]
    square = lambda pt: [[pt[0] + np.cos(th), pt[1] + np.sin(th)] for th in thetas]

    LLrep = []
    UTMrep = []
    dists = []
    with open(clicks_csv) as clicks:
        reader = csv.reader(clicks)
        next(reader)
        for line in reader:
            # break down line
            u = utm.from_latlon(float(line[0]), float(line[1]))
            ll = '(' + ','.join(line[:2]) + ')'
            print('lat/lon click location: ', ll)
            print('    utm conversion: ', u)
            tag = int(line[-1][-1])
            UTMrep.append([u[0], u[1]])  # clicks in UTM coordinates, meter base unit
    UTMrep.append([u[0]-0.1, u[1]-0.1])
    UTMrep.append([u[0]+0.1, u[1]+0.1])
    UTMrep.append([u[0]+0.1, u[1]-0.1])
    UTMrep.append([u[0]-0.1, u[1]+0.1])
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
                waypoints += square(pt)
            else:
                tmp += square(pt)


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
                waypoints += square(pt)
            else:
                tmp += square(pt)
        tmp = np.array(tmp)
        if len(tmp) != 0:
            hull = ConvexHull(tmp)
            pts = tmp[hull.vertices]
            plt.plot(pts[:,0], pts[:,1], 'o', markerfacecolor='g', markeredgecolor="k", markersize=6,)
            waypoints += list(pts)

    waypoints = np.array(waypoints)
    plt.title(f"Estimated number of clusters: {n_clusters_}")
# plt.show()
    for i, pti in enumerate(waypoints):
        tmp = []
        for j, ptj in enumerate(waypoints[i+1:]):
            tmp.append(np.linalg.norm(pti-ptj))
        dists.append(tmp)

    places = [i for i in range(len(waypoints))]
# print(places)
    out = tsp(places, dists)
    spt = out[0]
    plt.plot(waypoints[spt,0], waypoints[spt,1], 'o', markerfacecolor='r', markeredgecolor='k', markersize=14)
    for pt in out[1:]:
        plt.plot([waypoints[spt,0], waypoints[pt,0]], [waypoints[spt,1], waypoints[pt,1]], 'k')
        plt.plot(waypoints[pt,0], waypoints[pt,1], 'o', markerfacecolor='b', markeredgecolor='b', markersize=4)
        spt = pt
    plt.show()

    plan = []
    for pt in waypoints[out]:
        plan.append(list(utm.to_latlon(pt[0], pt[1], u[-2], u[-1])))
    plan = np.array(plan)

    generate_csv_from_plan(plan)

    """
    kml=simplekml.Kml()
    for i in out:
        print(plan[i])
        kml.newpoint(name=str(places[i]), coords=[plan[i]])
    kml.save(plan_kml)
    """


if __name__ == "__main__":
    build_dji_plan(generate=True)
