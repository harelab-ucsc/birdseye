from scipy.spatial.transform import Rotation as R
import numpy as np


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
