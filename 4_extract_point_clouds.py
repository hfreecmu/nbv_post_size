import argparse
import os
import json
import yaml
import numpy as np
import cv2
import open3d
from scipy.spatial.transform import Rotation

from nbv_utils import extract_point_cloud, get_paths, create_point_cloud
from nbv_utils import write_pickle

# def get_paths(data_dir):
#     inds_path = os.path.join(data_dir, 'indices.json')
#     indices = read_json(inds_path)

#     left_dir = os.path.join(data_dir, 'rect_images', 'left')
#     disparities_dir = os.path.join(data_dir, 'disparities')
#     #use right camera info for baseline
#     camera_info_dir = os.path.join(data_dir, 'camera_info', 'right')
#     transform_dir = os.path.join(data_dir, 'ee_states')

#     left_paths = []
#     disparity_paths = []
#     camera_info_paths = []
#     transform_paths = []

#     for filename in os.listdir(left_dir):
#         if not filename.endswith('.png'):
#             continue

#         index = int(filename.split('.png')[0])

#         if not index in indices:
#             continue

#         left_path = os.path.join(left_dir, filename)
#         disparity_path = os.path.join(disparities_dir, filename.replace('.png', '.npy'))
#         camera_info_path = os.path.join(camera_info_dir, filename.replace('.png', '.yml'))
#         transform_path = os.path.join(transform_dir, filename.replace('.png', '.yml'))

#         if not os.path.exists(disparity_path):
#             raise RuntimeError('No disparity for: ' + disparity_path)
        
#         if not os.path.exists(camera_info_path):
#             raise RuntimeError('No camera_info_path for: ' + camera_info_path)
        
#         if not os.path.exists(transform_path):
#             raise RuntimeError('No transform_path for: ' + transform_path)
        
#         left_paths.append(left_path)
#         disparity_paths.append(disparity_path)
#         camera_info_paths.append(camera_info_path)
#         transform_paths.append(transform_path)

#     return left_paths, disparity_paths, camera_info_paths, transform_paths

# def get_intrinsics(camera_info):
#     P = camera_info['projection_matrix']['data']
#     f_norm = P[0]
#     baseline = P[3] / P[0]
#     cx = P[2]
#     cy = P[6]
#     intrinsics = (baseline, f_norm, cx, cy)

#     return intrinsics

# def bilateral_filter(disparity, intrinsics, args):
#     baseline, f_norm, _, _ = intrinsics
#     stub = -baseline / disparity
#     z = stub * f_norm
#     z_new = cv2.bilateralFilter(z, args.bilateral_d, args.bilateral_sc, args.bilateral_ss)

#     stub_new = z_new / f_norm
#     disparity_new = -baseline / stub_new

#     return disparity_new

# def extract_depth_discontuinities(disparity, intrinsics, args):
#     baseline, f_norm, _, _ = intrinsics
#     stub = -baseline / disparity
#     z = stub * f_norm

#     element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dilation = cv2.dilate(z, element)
#     erosion = cv2.erode(z, element)

#     dilation -= z
#     erosion = z - erosion

#     max_image = np.max((dilation, erosion), axis=0)

#     if args.disc_use_rat:
#         ratio_image = max_image / z
#         _, discontinuity_map = cv2.threshold(ratio_image, args.disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
#     else:
#         _, discontinuity_map = cv2.threshold(max_image, args.disc_dist_thresh, 1.0, cv2.THRESH_BINARY)

#     element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)

#     return discontinuity_map

# def compute_points(disparity, intrinsics):
#     baseline, f_norm, cx, cy = intrinsics
#     stub = -baseline / disparity #*0.965

#     x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

#     x = stub * (x_pts - cx)
#     y = stub * (y_pts - cy)
#     z = stub*f_norm

#     points = np.stack((x, y, z), axis=2)

#     return points

# def create_point_cloud(cloud_path, points, colors, normals=None, estimate_normals=False):
#     cloud = open3d.geometry.PointCloud()
#     cloud.points = open3d.utility.Vector3dVector(points)
#     cloud.colors = open3d.utility.Vector3dVector(colors)

#     if normals is not None:
#         cloud.normals = open3d.utility.Vector3dVector(normals)
#     elif estimate_normals:
#         cloud.estimate_normals(
#             search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

#     open3d.io.write_point_cloud(
#         cloud_path,
#         cloud
#     ) 

# def get_transform(transform_path):
#     transform_dict = read_yaml(transform_path)
#     quat = transform_dict['quaternion']
#     trans = transform_dict['translation']

#     q = [quat['qx'], quat['qy'], quat['qz'], quat['qw']]
#     t = [trans['x'], trans['y'], trans['z']]

#     R = Rotation.from_quat(q).as_matrix()
#     t = np.array(t)

#     return R, t

# def extract_point_cloud(cloud_dir, world_dir, left_path, disparity_path, 
#                         camera_info_path, transform_path, args):
#     camera_info = read_yaml(camera_info_path)
#     intrinsics = get_intrinsics(camera_info)

#     disparity = np.load(disparity_path)

#     if np.min(disparity) <= 0:
#         print('WARNING: invalid disparities: ', (disparity <= 0).flatten().sum(), disparity_path)
#         disparity[disparity <= 0] = 1e-6

#     im = cv2.imread(left_path)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#     if args.bilateral_filter:
#         disparity = bilateral_filter(disparity, intrinsics, args)
    
#     discontinuity_map = extract_depth_discontuinities(disparity, intrinsics, args)
#     points = compute_points(disparity, intrinsics)
#     colors = im.astype(float) / 255

#     nan_inds = np.where(discontinuity_map > 0) 
#     points[nan_inds] = np.nan
#     colors[nan_inds] = np.nan

#     if args.z_only:
#         nan_inds = np.where(points[:, :, 2] > args.max_dist)
#     else:
#         nan_inds = np.where(np.linalg.norm(points, axis=2) > args.max_dist)
#     points[nan_inds] = np.nan
#     colors[nan_inds] = np.nan

#     basename = os.path.basename(left_path).split('.png')[0]
#     cloud_path = os.path.join(cloud_dir, basename + '.pcd')

#     create_point_cloud(cloud_path, points.reshape((-1, 3)), colors.reshape((-1, 3)))

#     np_path =  os.path.join(cloud_dir, basename + '.npy')
#     np.save(np_path, points)

#     R, t = get_transform(transform_path)
#     world_points = ((R @ points.reshape((-1, 3)).T).T + t).reshape(points.shape)

#     world_path = os.path.join(world_dir, basename + '.pcd')
#     create_point_cloud(world_path, world_points.reshape((-1, 3)), colors.reshape((-1, 3)))

#     np_path =  os.path.join(world_dir, basename + '.npy')
#     np.save(np_path, world_points)

def extract_point_clouds(data_dir, args):
    path_ret = get_paths(data_dir, indices=None)
    _, left_paths, disparity_paths, camera_info_paths, transform_paths, _ = path_ret
    assert len(left_paths) == len(disparity_paths)
    assert len(left_paths) == len(camera_info_paths)
    assert len(left_paths) == len(transform_paths)

    cloud_dir = os.path.join(data_dir, 'clouds_camera')
    if not os.path.exists(cloud_dir):
        os.mkdir(cloud_dir)

    world_dir = os.path.join(data_dir, 'clouds_world')
    if not os.path.exists(world_dir):
        os.mkdir(world_dir)

    for i in range(len(left_paths)):
        left_path = left_paths[i]
        disparity_path = disparity_paths[i]
        camera_info_path = camera_info_paths[i]
        transform_path = transform_paths[i]


        points, world_points, colors, R, t, discon_map, points_orig, colors_orig = extract_point_cloud(left_path, disparity_path,
                                                                                                       camera_info_path, transform_path,
                                                                                                       args, include_discon=True, 
                                                                                                       include_orig=True)
        basename = os.path.basename(left_path).split('.png')[0]
        cloud_path = os.path.join(cloud_dir, basename + '.pcd')
        create_point_cloud(cloud_path, points.reshape((-1, 3)), colors.reshape((-1, 3)))
        np_path =  os.path.join(cloud_dir, basename + '.npy')
        np.save(np_path, points)

        world_path = os.path.join(world_dir, basename + '.pcd')
        create_point_cloud(world_path, world_points.reshape((-1, 3)), colors.reshape((-1, 3)))
        np_path =  os.path.join(world_dir, basename + '.npy')
        np.save(np_path, world_points)

        discon_path = os.path.join(cloud_dir, basename + '_discon.png')
        discon_im = np.zeros(world_points.shape[0:2], dtype=np.uint8)
        discon_im[discon_map > 0] = 255
        cv2.imwrite(discon_path, discon_im)

        orig_path = os.path.join(cloud_dir, basename + '_orig.pcd')
        create_point_cloud(orig_path, points_orig.reshape((-1, 3)), colors_orig.reshape((-1, 3)))

        transform_path = os.path.join(cloud_dir, basename + '.pkl')
        write_pickle(transform_path, [R, t])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)

    parser.add_argument('--bilateral_filter', action='store_false')
    parser.add_argument('--bilateral_d', type=int, default=9)
    parser.add_argument('--bilateral_sc', type=float, default=0.03)
    parser.add_argument('--bilateral_ss', type=float, default=4.5)

    parser.add_argument('--disc_use_rat', action='store_false')
    parser.add_argument('--disc_rat_thresh', type=float, default=0.004)
    parser.add_argument('--disc_dist_thresh', type=float, default=0.001)

    parser.add_argument('--z_only', action='store_true')
    parser.add_argument('--max_dist', type=float, default=0.6)
    

    args = parser.parse_args()
    return args

#python3 4_extract_point_clouds.py --data_dir=$DATA_DIR
if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    extract_point_clouds(data_dir, args)
