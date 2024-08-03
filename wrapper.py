import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from utils import *
from prediction import predict_from_checkpoint
from make_video import *

def main():

    # Caluclating the transformation from LiDAR to camera
    lidar_to_cam = lidar_to_cam_projectn()

    img_save = os.path.join('Results', 'projected_clouds')
    pointcloud_save = os.path.join('Results', 'painted_clouds')
    make_directory(img_save)
    make_directory(pointcloud_save)

    source_directory = os.path.join('Data', '3d_data')  
    target_directory = os.path.join('Data', 'pcd_data') 

    convert_all_bin_to_pcd(source_directory, target_directory)
    print("Converted all bin files to pcd files")

    rgb_folder_path = os.path.join('Data', '2d_data')
    pcd_folder_path = os.path.join('Data', 'pcd_data')
    img_list= os.listdir(rgb_folder_path)
    img_list.sort()
    pcd_list = os.listdir(pcd_folder_path)
    pcd_list.sort()

    img_paths = []
    pcd_paths = []

    for a in range(len(img_list)):
        img_paths.append(os.path.join(rgb_folder_path, img_list[a]))
        pcd_paths.append(os.path.join(pcd_folder_path, pcd_list[a]))

    for i in range(len(img_list)):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        pcd = o3d.io.read_point_cloud(pcd_paths[i])
        pcd_arr = np.asarray(pcd.points)

        # removing points from the point cloud that are behind the camera
        idx = pcd_arr[:,0] >= 0  
        pcd_arr = pcd_arr[idx]
        pts_2D, depth, pts_3D_img = projecting_lidar_on_img(lidar_to_cam, pcd_arr, (img.shape[1], img.shape[0]))

        # DeepLabV3+ pretrained model
        checkpoint_path = '/home/shreya/Projects/lidar_segmentation/checkpoint/best_deeplabv3plus_resnet101_cityscapes_os16.pth'
        
        pred, semantic_img = predict_from_checkpoint(checkpoint_path, img_path)
        # print("Predicted semantic segmentation of RGB image")

        point_cloud_clr = np.zeros((pts_3D_img.shape[0],3), dtype=np.float32)
        iimg = img.copy()
        for j in range(pts_2D.shape[0]):
            if j >= 0:

                x, y = np.int32(pts_2D[j, 0]), np.int32(pts_2D[j, 1])
                
                class_color = np.float64(semantic_img[y, x])
                cv2.circle(iimg, (x,y), 2, color=tuple(class_color), thickness=1)
                point_cloud_clr[j] = class_color/255.0

        stacked_img = np.vstack((img,iimg))
        cv2.imwrite(img_save + "/" + str(i) + ".png",stacked_img)
        
        semantic_point_cloud = np.hstack((pts_3D_img[:,:3], point_cloud_clr))
        visualization(semantic_point_cloud, i, pointcloud_save)

    video_file = "video.mp4"
    make_video(10, img_save, video_file)

if __name__ == '__main__':
    main()