#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utilities import *
from scipy.spatial.transform import Rotation as R


def arrow3d(ax, length=1, width=0.05, head=0.2, headwidth=1,
                theta_x=0, theta_z=0, offset=(0,0,0), rotation=np.eye(3), **kw):
    w = width
    h = head
    hw = headwidth
    theta_x = np.deg2rad(theta_x)
    theta_z = np.deg2rad(theta_z)

    a = [[0,0],[w,0],[w,(1-h)*length],[hw*w,(1-h)*length],[0,length]]
    a = np.array(a)

    r, theta = np.meshgrid(a[:,0], np.linspace(0,2*np.pi,30))
    z = np.tile(a[:,1],r.shape[0]).reshape(r.shape)
    x = r*np.sin(theta)
    y = r*np.cos(theta)

    #prepare a rotation matrix for each axis
    rot_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x) ],
                      [0,np.sin(theta_x) ,np.cos(theta_x) ]])
    rot_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0 ],
                      [np.sin(theta_z) ,np.cos(theta_z),0 ],[0,0,1]])

    b1 = np.dot(rot_x, np.c_[x.flatten(),y.flatten(),z.flatten()].T)
    b2 = np.dot(rot_z, b1)
    b3 = np.dot(rotation, b2)
    b4 = b3.T+np.array(offset)
    x = b4[:,0].reshape(r.shape);
    y = b4[:,1].reshape(r.shape);
    z = b4[:,2].reshape(r.shape);
    ax.plot_surface(x,y,z, **kw)


def plotTransform(ax, T, labels=['Camera x-axis','Camera y-axis','Camera z-axis'], colors=['r', 'g', 'b']):
    #Given a homogeneous transform, plot the triad:
    roll, pitch, yaw = R.from_matrix(T[0:3,0:3]).as_euler('xyz')
    # print(f'    roll, pitch, yaw: {roll}, {pitch}, {yaw}')
    x, y, z = T[:3,3]
    # print(f'    x, y, z: {x}, {y}, {z}')
    plotTriad(ax, x, y, z, roll, pitch, yaw, colors=colors, labels=labels)
    return roll, pitch, yaw, x, y, z


def plotTriad(ax, x, y, z, roll, pitch, yaw, colors, labels):
    """
    Plots a coordinate triad with X-forward, Y-right, Z-down convention.

    Parameters:
        ax      - Matplotlib 3D axis.
        x, y, z - Origin of the triad.
        roll    - Rotation about the X-axis (forward).
        pitch   - Rotation about the Y-axis (right).
        yaw     - Rotation about the Z-axis (down).
        colors  - List of colors for X, Y, Z axes.
        labels  - List of labels for X, Y, Z axes.
    """
    # Triad length
    L = 0.8

    # Rotation matrices (X-forward, Y-right, Z-down)
    r = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    # r = np.array([[0,1,0],[1,0,0],[0,0,1]])@r

    # Triad axes in local frame (before rotation)
    x_axis = np.array([L, 0, 0])  # Forward
    y_axis = np.array([0, L, 0])  # Right
    z_axis = np.array([0, 0, L])  # Down

    # Rotate axes (yaw → pitch → roll)
    x_axis = r @ x_axis
    y_axis = r @ y_axis
    z_axis = r @ z_axis

    # Draw triad
    ax.quiver(x, y, z, x_axis[0], x_axis[1], x_axis[2], color=colors[0], label=labels[0])
    ax.quiver(x, y, z, y_axis[0], y_axis[1], y_axis[2], color=colors[1], label=labels[1])
    ax.quiver(x, y, z, z_axis[0], z_axis[1], z_axis[2], color=colors[2], label=labels[2])


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    radius = 5
    for i in np.linspace(0,2*np.pi, 10):
        plotTriad(ax, [radius*np.sin(i),radius*np.cos(i),10], [0,np.pi,i])

    plotTriad(ax, [0,0,0], [0,0,0])

    ax.set_xlim(-10,10)
    ax.set_xlabel('X')

    ax.set_ylim(-10,10)
    ax.set_ylabel('Y')

    ax.set_zlim(0,20)
    ax.set_zlabel('Z')
    plt.show()
