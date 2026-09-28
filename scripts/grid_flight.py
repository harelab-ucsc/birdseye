#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import pdb
import argparse
import cv2
import os
import csv
import utm

from scipy.spatial import ConvexHull


class gridPadder():
    def __init__(self, kwargs):
        self.clicks_csv = kwargs.pop('clicks_csv', None)
        self.plan_csv = kwargs.pop('plan_csv', None)
        self.grid_dim = kwargs.pop('grid_dim', 4)

        self.data = None
        self.zone_num = None
        self.zone_let = None
        self.grid = None
        self.copy = None
        self.hull = None
        self.val = None
        self.waypoints = None


    def csv_read(self):
        print(f'Reading clicks CSV file: {self.clicks_csv}...')
        self.data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                # self.get_logger().info(f'{line}')
                u = utm.from_latlon(float(line[0]), float(line[1]))  # returns easting, northing, zone number, zone letter
                if self.zone_let is None or self.zone_num is None:
                    self.zone_num = u[2]
                    self.zone_let = u[3]
                tag = int(line[-1][-1])
                self.data.append([u[0], u[1], float(line[2]), float(line[3]), tag])
        return self.data


    def make_grid(self):
        self.grid = []
        if self.copy is None:
            self.copy = np.array(self.data)
            self.copy = self.copy[:,:2]
        mins = self.copy.min(axis=0)
        maxs = self.copy.max(axis=0)
        x_steps = (maxs[0]-mins[0])//self.grid_dim
        y_steps = (maxs[1]-mins[1])//self.grid_dim
        x = np.linspace(mins[0], maxs[0], int(x_steps))
        y = np.linspace(mins[1], maxs[1], int(y_steps))

        for i in x:
            for j in y:
                self.grid.append([i,j])
        # plt.scatter(np.array(self.grid)[:,0], np.array(self.grid)[:,1])
        # plt.show()


    def getHull(self):
        if self.copy is None:
            self.copy = np.array(self.data)
            self.copy = self.copy[:,:2]
        self.hull = ConvexHull(self.copy)


    def inFrameCheck(self):
        n = self.hull.vertices.shape[0]
        frame = np.array(self.copy[self.hull.vertices, :])
        # print('    inQuadCheck frame: \n', frame)
        val = []
        ring = lambda y: [ (x + 1) % y for x in range(y)]
        det = lambda x,y: x[0]*y[1] - [x[1]*y[0]]
        tmp1 = frame[ring(n)] - frame

        for v in self.grid:
            v = np.array(v)
            tmp2 = v - frame
            tmp = [int(det(tmp1[i], tmp2[i])) >= 0 for i in range(n)]
            val.append(sum(tmp))
        self.val = [True if i == n else False for i in val]


    def make_waypoints(self):
        self.waypoints = []
        self.waypoints += self.copy.tolist()
        self.waypoints += np.array(self.grid)[self.val].tolist()
        # plt.scatter(np.array(self.waypoints)[:,0], np.array(self.waypoints)[:,1])
        # plt.show()


    def export_csv(self, mode='w'):
        print(f'Saving grid-padded waypoints csv: {self.plan_csv}')

        with open(self.plan_csv, mode, newline='') as f:
            writer = csv.writer(f)
            for line in self.waypoints:
                tmp = list(utm.to_latlon(line[0], line[1], self.zone_num, self. zone_let))
                if mode == 'w':
                    writer.writerow(tmp + ['wgs84_alt', 'msl_alt', 2, 1.0])
                elif mode == 'a':
                    writer.writerow(tmp + ['wgs84_alt', 'msl_alt', 2, 0.0])


def main():
    parser = argparse.ArgumentParser(description="Fix image timestamps in a ROS2 bag file using INS messages.")
    parser.add_argument("-c", "--clicks_csv", help="Path to the input geotag csv file")
    parser.add_argument("-p", "--plan_csv", help="Path to the input geotag csv file")
    # parser.add_argument("ds_dir",  help="Path to the directory to save images/ and poses.json to")
    # parser.add_argument("image_topic", help="Image topic name (e.g., /camera/image_raw)")

    args = vars(parser.parse_args())

    obj = gridPadder(args)
    obj.csv_read()
    obj.make_grid()
    obj.getHull()
    obj.inFrameCheck()
    obj.make_waypoints()
    obj.export_csv()


if __name__ == "__main__":
    main()
