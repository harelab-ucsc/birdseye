#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R

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


def arrow3d(ax, length=1, width=0.05, head=0.2, headwidth=2,
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


def plotTransform(ax, T):
    #Given a homogeneous transform, plot the triad:
    euler = dcm2euler(T[0:3,0:3])
    xyz = T[0:3,3]
    plotTriad(ax, xyz, euler)

def plotTriad(ax, xyz, rpy):

    
    #default: length along z axis
    # Handle rotate about x, call that roll:
    
    #Make an arbitrary rotation matrix for each of roll, pitch, yaw: (x forward, y left, z up)

    theta_x = rpy[0]
    theta_y = rpy[1]
    theta_z = rpy[2]
    
    rot_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x) ],
                      [0,np.sin(theta_x) ,np.cos(theta_x) ]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                      [0,1,0],
                      [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0 ],
                     [np.sin(theta_z) ,np.cos(theta_z),0 ],[0,0,1]])
    full_rot = rot_x@rot_y@rot_z
            
                     
    arrow3d(ax, length=1, color="blue", offset=xyz, rotation=full_rot)
    arrow3d(ax, length=1, theta_x=-90, color="limegreen", offset=xyz, rotation=full_rot) 
    arrow3d(ax, length=1, theta_x=90, theta_z=90, color="crimson", offset=xyz, rotation=full_rot)


    
#arrow3d(ax, length=2, width=0.02, head=0.1, headwidth=1.5, offset=[1,1,0], 
#        theta_x=40,  color="crimson")

#arrow3d(ax, length=1.4, width=0.03, head=0.15, headwidth=1.8, offset=[1,0.1,0], 
#        theta_x=-60, theta_z = 60,  color="limegreen")

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
