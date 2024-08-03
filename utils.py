import numpy as np
import open3d
import cv2
import os
import matplotlib.pyplot as plt
import struct
import copy
import shutil


# To create a directory
def make_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

# To read calibration data
def read_calib_data():
    filepath = "Data/calibration/calib.txt"
    calib_data = {}
    with open(filepath, "r") as file:
        for row in file:
            key, value = row.split(':')
            values = np.array([float(x) for x in value.split()])
            if key == "P":
                calib_data[key] = values.reshape((3, 4))
            elif key == "K":
                calib_data[key] = values.reshape((3, 3))
            elif key == "R0":
                calib_data[key] = values.reshape((3, 3))
            elif key == "Tr_cam_to_lidar":
                calib_data[key] = values.reshape((3, 4))
            elif key == "D":
                calib_data[key] = values.reshape((1, 5))
    return calib_data["P"], calib_data["K"], calib_data["R0"], calib_data["Tr_cam_to_lidar"], calib_data["D"]

# To calculate the projection matrix
def lidar_to_cam_projectn():
    P, K, R, Tr_cam_to_lidar, D = read_calib_data()

    R_cam_to_lidar = Tr_cam_to_lidar[:3,:3].reshape(3,3)
    t_cam_to_lidar = Tr_cam_to_lidar[:3,3].reshape(3,1)

    R_cam_to_lidar_inv = np.linalg.inv(R_cam_to_lidar)
    t_new = -np.dot(R_cam_to_lidar_inv , t_cam_to_lidar)
    Tr_lidar_to_cam = np.vstack((np.hstack((R_cam_to_lidar_inv, t_new)), np.array([0., 0., 0., 1.])))
    R_rect = np.eye(4)
    R_rect[:3, :3] = R.reshape(3, 3)
    P_ = P.reshape((3, 4))
    proj_mat = P_ @ R_rect  @ Tr_lidar_to_cam
    return proj_mat

# To check if the points are inside the image frame
def inside_point_idx(pts_2d, size): # check if the points are inside the image frame
    return ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < size[0]) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < size[1]))

# To project the lidar points back on the image
def projecting_lidar_on_img(P, lidar_pts, size):

    n = lidar_pts.shape[0]
    pts_3d =  np.hstack((lidar_pts, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d, P.T)
    depth = pts_3d[:,2] # last column of the the projected points 
    depth[depth==0] = -1e-6

    #normalize 3d points
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d = pts_2d[:, :2]

    inliers_idx = inside_point_idx(pts_2d, size)
    return pts_2d[inliers_idx], depth[inliers_idx], lidar_pts[inliers_idx]

# To convert bin files to pcd files
def bin_to_pcd(bin_path, pcd_path):
    size_float = 4
    list_pcd = []
    with open(bin_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    open3d.io.write_point_cloud(pcd_path, pcd)


def convert_all_bin_to_pcd(folder_path, save_path):
    bin_list= os.listdir(folder_path)
    for i in range(len(bin_list)):
        # print("count", i)
        bin_path = os.path.join(folder_path,bin_list[i])
        bin_to_pcd(bin_path, save_path + "/" + str(i) + ".pcd")

# To visualize the point cloud
def visualization(pointcloud, count, save_path):

    xyz = pointcloud[:, 0:3]
    semantics = pointcloud[:, 3:]

    #Initialize Open3D visualizer    
    visualizer = open3d.visualization.Visualizer()
    pcd = open3d.geometry.PointCloud()
    visualizer.add_geometry(pcd)
    
    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(semantics)

    open3d.io.write_point_cloud(save_path + "/" + str(count) + ".pcd",pcd)