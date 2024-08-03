import open3d as o3d
import numpy as np
import copy
import os

# Function to perform point to point ICP
def point_to_point_icp(source, target, threshold, transformation):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    return reg_p2p.transformation


if __name__ == "__main__":
    base_path = './Results/painted_clouds/'
    pcd_data = os.listdir(base_path)
    pcd_data.sort()
    transformation_list = []
    accumulated_transformation = np.identity(4)

    for i in range(len(pcd_data)-1):
        source = o3d.io.read_point_cloud(base_path+pcd_data[i])
        target = o3d.io.read_point_cloud(base_path+pcd_data[i+1])
        threshold = 0.2
        trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                [-0.139, 0.967, -0.215, 0.7],
                                [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

        transformation = point_to_point_icp(source, target, threshold, trans_init)
        accumulated_transformation = np.dot(transformation, accumulated_transformation)
        transformation_list.append(accumulated_transformation)

    if base_path=='./Results/painted_clouds':
        o3d.io.write_point_cloud('pointcloud.pcd',source)
    else:
        o3d.io.write_point_cloud('painted_pointcloud.pcd',source)

    o3d.visualization.draw_geometries([source])