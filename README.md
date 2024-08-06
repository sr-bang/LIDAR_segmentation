# LiDAR Semantic Segmentation
To create a pipeline that integrates data from LiDAR and Camera to produce a semantically annotated point cloud.

## Dataset
Download the rectified stereo camera images and Velodyne sensor data from the KITTI-360 dataset. Save data from any of the 9 sequences in Data/2d_data and Data/3d_ data folders respectively. To download them run following shell script. 
1. ```bash download_2d_perspective.sh```
2. ```bash download_2d_perspective.sh```
3. For [Camera intrinsics and extrinsics](https://www.cvlibs.net/datasets/kitti-360/index.php)


## Checkpoint
For the semantic segmentation of RGB images, download pretrained DeepLabV3+ weights and save them in Checkpoint folder.
[Download](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing)

## Code
To get the semantics RGB and transfer labels to point cloud data:
```
python3 wrapper.py
```
To register the semantics point cloud data:
```
python3 registration.py
```

## Result

![pt_cloud_registration](https://github.com/user-attachments/assets/b5c543a0-39f9-4416-abc2-62d2be0ee5f1)

https://github.com/user-attachments/assets/30c129fa-4eb8-4788-9a21-1bf376e881d3
