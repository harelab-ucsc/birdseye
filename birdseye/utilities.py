#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys
import cv2


# sine helper function
def s(ang):
    value = np.sin(ang)
    return value


# cosine helper function
def c(ang):
    value = np.cos(ang)
    return value


def dcm2euler(dcm):
    # yaw calculations
    yaw = np.arctan2(dcm[0][1], dcm[0][0])

    # pitch calculations
    # the min, max ensures that the value remains within [-1,1]
    boundedCell = min(1,max(-1,dcm[0][2]))
    pitch = -np.arcsin(boundedCell)

    # roll calculations
    roll = np.arctan2(dcm[1][2], dcm[2][2])
    euler = [roll, pitch, yaw] #be consistent in the ordering of xyz axes

    return euler


# Create the direction cosine matrix, R, from the euler angles
# The DCM goes from inertial vectors into the body frame
def euler2dcm(roll, pitch, yaw): #about axes ordered x,y,z
    dcm = [[(c(pitch) * c(yaw)), (c(pitch) * s(yaw)), -(s(pitch))],
           [((s(roll) * s(pitch) * c(yaw)) - (c(roll) * s(yaw))), ((s(roll) * s(pitch) * s(yaw)) + (c(roll) * c(yaw))), (s(roll) * c(pitch))],
           [((c(roll) * s(pitch) * c(yaw)) + (s(roll) * s(yaw))), ((c(roll) * s(pitch) * s(yaw)) - (s(roll) * c(yaw))), (c(roll) * c(pitch))]]
    return dcm


def euler2quat(roll, pitch, yaw, degrees=False):
    euler = np.array([roll, pitch, yaw])
    quat = R.from_euler('xyz', euler, degrees).as_quat()
    return quat


def quat2euler(qx, qy, qz, qw, degrees=False):
    euler = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees)
    return euler


def dimIterProd(mtx_dims):
    tmp = 1
    for d in mtx_dims:
        tmp *= d
    return tmp


def array_flatten(mtx, mtx_dims):
    """ mtx is m x n (rectangular, nonsparse) array """
    val = []
    for i in range(mtx_dims[0]):
        for j in range(mtx_dims[1]):
            val.append(mtx[i][j])
    return val


def array_expand(mtx, mtx_dims):
    """ mtx is mn x 1 (m*n dimensional vector) """
    tmp = []
    tmps = []
    row = 0
    for i, elem in enumerate(mtx):
        tmp.append(elem)
        if i == row*mtx_dims[1] + (mtx_dims[1]-1):
            row += 1
            tmps.append(tmp)
            tmp = []
    return tmps


def matrix_list_converter(mtx : list, mtx_dims):
    if len(mtx) == dimIterProd(mtx_dims):
        # print('flat')
        return array_expand(mtx, mtx_dims)
    else:
        # print('not flat')
        return array_flatten(mtx, mtx_dims)


def string_list_converter(foo):
    if isinstance(foo, str):
        if foo != 'None':
            val = []
            tmp = foo.split('[')[1]
            tmp = tmp.split(']')[0]
            for item in tmp.split(', '):
                if item != '':
                    val.append(float(item))
            return val
        else:
            return None
    elif isinstance(foo, list):
        val = '['
        for item in foo:
            val += str(item)
        val += ']'
        return val
    else:
        print('oops')


def poseRowToTransform(pose, rpy=None):
    #Given a row from the db, produce a 4x4 homogeneous transform
    #Return as a 4x4 nparray
    if rpy is None:
        rpy = quat2euler(pose[3], pose[4], pose[5], pose[6])

    rot = euler2dcm(rpy[0], rpy[1], rpy[2])
    translate = [[pose[0]],[pose[1]],[pose[2]]]
    T = np.hstack([rot, translate])
    T = np.vstack([T, [0,0,0,1]])

    return T


def makePoseMatrix(trans, rot):
    # print(trans, rot)
    if len(rot) == 4:
        rot = R.from_quat(rot).as_matrix()
    if type(trans) == list:
        print('trans is a list')
        trans = np.array([trans]).T
    else:
        trans = np.expand_dims(trans, axis=0).T
        # print(trans)

    # make a 4x4 pose matrix
    pose = np.concatenate((rot, trans), axis=1)
    pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
    return pose, rot, trans


def makeAPose(x, y, z, roll, pitch, yaw):
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    tra = np.array([x, y, z])
    return makePoseMatrix(tra, rot)


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
    x = b4[:,0].reshape(r.shape)
    y = b4[:,1].reshape(r.shape)
    z = b4[:,2].reshape(r.shape)
    ax.plot_surface(x,y,z, **kw)


def plotTransform(ax, T, labels=['Camera x-axis','Camera y-axis','Camera z-axis'], colors=['r', 'g', 'b']):
    #Given a homogeneous transform, plot the triad:
    roll, pitch, yaw = dcm2euler(T[0:3,0:3])
    x, y, z = T[0:3,3]
    plotTriad(ax, x, y, z, roll, pitch, yaw, colors=colors, labels=labels)


def plotTriad(ax, x, y, z, roll, pitch, yaw, colors, labels):
    # default: length along z axis
    # Handle rotate about x, call that roll:

    # Make an arbitrary rotation matrix for each of roll, pitch, yaw: (x forward, y left, z up)
    # Triad length
		L = 0.8

		# Rotation matrices
		R_roll = np.array([[1, 0, 0],
						   [0, np.cos(roll), -np.sin(roll)],
						   [0, np.sin(roll), np.cos(roll)]])

		R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
						    [0, 1, 0],
						    [-np.sin(pitch), 0, np.cos(pitch)]])

		R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
						  [np.sin(yaw), np.cos(yaw), 0],
						  [0, 0, 1]])

		# Triad axes
		x_axis = np.array([L, 0, 0])
		y_axis = np.array([0, L, 0])
		z_axis = np.array([0, 0, L])

		# Rotate axes according to roll, pitch, yaw
		x_axis = np.dot(R_yaw, np.dot(R_pitch, np.dot(R_roll, x_axis)))
		y_axis = np.dot(R_yaw, np.dot(R_pitch, np.dot(R_roll, y_axis)))
		z_axis = np.dot(R_yaw, np.dot(R_pitch, np.dot(R_roll, z_axis)))

		# Draw triad
		ax.quiver(x, y, z, x_axis[0], x_axis[1], x_axis[2], color=colors[0], label=labels[0])
		ax.quiver(x, y, z, y_axis[0], y_axis[1], y_axis[2], color=colors[1], label=labels[1])
		ax.quiver(x, y, z, z_axis[0], z_axis[1], z_axis[2], color=colors[2], label=labels[2])
		# ax.legend()


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

