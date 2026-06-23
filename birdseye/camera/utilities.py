#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
import cv2
# import apriltag


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
    # pdb.set_trace()
    # print('mtx: ', mtx)
    # print('mtx_dims: ', mtx_dims)

    if len(mtx) == dimIterProd(mtx_dims):
        # print('flat')
        return array_expand(mtx, mtx_dims)
    else:
        # print('not flat')
        return array_flatten(mtx, mtx_dims)


# def matrix_list_converter_test():
#     rng = np.random.default_rng()
#     ctrl = rng.multivariate_normal([0.0,0.0,0.0,0.0], np.eye(4), 3)
#     print('ctrl: \n', ctrl)
#     print(ctrl.shape)
#     go = matrix_list_converter(ctrl.tolist(), ctrl.shape)
#     assert dimIterProd(ctrl.shape) == len(go)
#     print('go: ', np.array(go), '\n')
#     go = matrix_list_converter(go, ctrl.shape)
#     print('go: ', go, '\n')
#     assert np.equal(np.array(go), ctrl).sum() == dimIterProd(ctrl.shape)
#     print('go: ', go, '\n')


def string_list_converter(foo):
    # print('foo: ', foo)
    if isinstance(foo, str):
        if foo != 'None':
            val = []
            tmp = foo.split('[')[1]
            # print(tmp)
            tmp = tmp.split(']')[0]
            # print(tmp)
            for item in tmp.split(', '):
                if item != '':
                    val.append(float(item))
            # print(val)
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


def chronCheck(sec, nsec, click, pattern):
    # print(f'chronCheck: patt: {pattern}')
    if pattern == '<':  # in the future
        if sec - click[0] < 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in future of {int(sec)}.{int(nsec)}')
            return True
        elif sec - click[0] == 0 and nsec - click[1] < 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in future of {int(sec)}.{int(nsec)}')
            return True
        else:
            return False

    elif pattern == '>':  # in the past
        if sec - click[0] > 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in past of {int(sec)}.{int(nsec)}')
            return True
        elif sec - click[0] == 0 and nsec - click[1] > 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in past of {int(sec)}.{int(nsec)}')
            return True
        else:
            return False

    else:
        print('error: pattern is not \'>\' or \'<\'')
        return False


def clickInView(sec, nsec, old_sec, old_nsec, click_px):
    # print('clickInView')
    valid = []
    if old_sec is None:
        return valid
    for i, click in enumerate(click_px):
        if chronCheck(old_sec, old_nsec, click, '<') and chronCheck(sec, nsec, click, '>'):
            valid.append(i)
            print(f'valid found: {click_px[i]}')
    return valid
