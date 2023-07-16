import os
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import open3d
import json
import pickle

###WARNING WARNING WARNING
#any changes made here may also have to go in nbv_utils in feature_assocation repo

#read json file
def read_json(path):
    with open(path, 'r') as f:
        json_to_read = json.loads(f.read())
    
    return json_to_read

#read yaml file
def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

#read pkl file
def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#write json file
def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

#write pkl file
def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

#get node id
def get_node_id(image_ind, seg_ind):
    return str(image_ind) + '_' + str(seg_ind)

#parse node id
def parse_node_id(node_id):
    image_ind, seg_ind = node_id.split('_')
    return int(image_ind), int(seg_ind)

#get R and t inverse
def get_rot_trans_inv(R, t):
    t_inv = -R.T @ t
    R_inv = R.T

    return R_inv, t_inv

#get intrinsics struck from dict
def get_intrinsics(camera_info):
    P = camera_info['projection_matrix']['data']
    f_norm = P[0]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]
    intrinsics = (baseline, f_norm, cx, cy)

    return intrinsics

#get transform from yml path
def get_transform(transform_path):
    transform_dict = read_yaml(transform_path)
    quat = transform_dict['quaternion']
    trans = transform_dict['translation']

    q = [quat['qx'], quat['qy'], quat['qz'], quat['qw']]
    t = [trans['x'], trans['y'], trans['z']]

    R = Rotation.from_quat(q).as_matrix()
    t = np.array(t)

    return R, t

#bilateral filter
def bilateral_filter(disparity, intrinsics, args):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm
    z_new = cv2.bilateralFilter(z, args.bilateral_d, args.bilateral_sc, args.bilateral_ss)

    stub_new = z_new / f_norm
    disparity_new = -baseline / stub_new

    return disparity_new

#extract depth discontinuities
def extract_depth_discontuinities(disparity, intrinsics, args):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(z, element)
    erosion = cv2.erode(z, element)

    dilation -= z
    erosion = z - erosion

    max_image = np.max((dilation, erosion), axis=0)

    if args.disc_use_rat:
        ratio_image = max_image / z
        _, discontinuity_map = cv2.threshold(ratio_image, args.disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
    else:
        _, discontinuity_map = cv2.threshold(max_image, args.disc_dist_thresh, 1.0, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)

    return discontinuity_map

#compute points using our method
def compute_points(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    stub = -baseline / disparity #*0.965

    x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

    x = stub * (x_pts - cx)
    y = stub * (y_pts - cy)
    z = stub*f_norm

    points = np.stack((x, y, z), axis=2)

    return points

#compute points using opencv - same as above
def compute_points_opencv(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    
    cm1 = np.array([[f_norm, 0, cx],
                   [0, f_norm, cy],
                   [0, 0, 1]])
    
    cm2 = np.array([[f_norm, 0, cx],
                   [0, f_norm, cy],
                   [0, 0, 1]])
    
    distortion1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    distortion2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    T = np.array([baseline, 0, 0])
    R = np.eye(3)


    res = cv2.stereoRectify(cm1, distortion1, cm2, distortion2, (1440, 1080), R, T)
    Q = res[4]

    points = cv2.reprojectImageTo3D(disparity, Q)

    return points

#save point cloud file
def create_point_cloud(cloud_path, points, colors, normals=None, estimate_normals=False):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)
    cloud.colors = open3d.utility.Vector3dVector(colors)

    if normals is not None:
        cloud.normals = open3d.utility.Vector3dVector(normals)
    elif estimate_normals:
        cloud.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    open3d.io.write_point_cloud(
        cloud_path,
        cloud
    ) 

#extract point cloud using filters
def extract_point_cloud(left_path, disparity_path, 
                        camera_info_path, transform_path, args,
                        include_discon=False):
    camera_info = read_yaml(camera_info_path)
    intrinsics = get_intrinsics(camera_info)

    disparity = np.load(disparity_path)
    if np.min(disparity) <= 0:
        print('WARNING: invalid disparities: ', (disparity <= 0).flatten().sum(), disparity_path)
        disparity[disparity <= 0] = 1e-6

    im = cv2.imread(left_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if args.bilateral_filter:
        disparity = bilateral_filter(disparity, intrinsics, args)
    
    discontinuity_map = extract_depth_discontuinities(disparity, intrinsics, args)
    points = compute_points(disparity, intrinsics)
    colors = im.astype(float) / 255

    nan_inds = np.where(discontinuity_map > 0) 
    points[nan_inds] = np.nan
    colors[nan_inds] = np.nan

    if args.z_only:
        nan_inds = np.where(points[:, :, 2] > args.max_dist)
    else:
        nan_inds = np.where(np.linalg.norm(points, axis=2) > args.max_dist)
    points[nan_inds] = np.nan
    colors[nan_inds] = np.nan

    R, t = get_transform(transform_path)
    world_points = ((R @ points.reshape((-1, 3)).T).T + t).reshape(points.shape)

    if not include_discon:
        return points, world_points, colors, R, t
    else:
        return points, world_points, colors, R, t, discontinuity_map

#get paths following directory structure
def get_paths(data_dir, indices, single=False, 
              use_filter_segs=False, include_cloud=False,
              use_filter_indices=False):
    left_dir = os.path.join(data_dir, 'rect_images', 'left')
    disparities_dir = os.path.join(data_dir, 'disparities')
    #use right camera info for baseline
    camera_info_dir = os.path.join(data_dir, 'camera_info', 'right')
    transform_dir = os.path.join(data_dir, 'ee_states')
    clouds_camera_dir = os.path.join(data_dir, 'clouds_camera')
    clouds_world_dir = os.path.join(data_dir, 'clouds_world')

    if not use_filter_segs:
        segmentation_dir = os.path.join(data_dir, 'segmentations')
    else:
        segmentation_dir = os.path.join(data_dir, 'segmentations_filtered')

    left_paths = []
    disparity_paths = []
    camera_info_paths = []
    transform_paths = []
    segmentation_paths = []
    clouds_camera_paths = []
    clouds_world_paths = []
    path_inds = []

    if indices is None:
        if not use_filter_indices:
            inds_path = os.path.join(data_dir, 'indices.json')
        else:
            inds_path = os.path.join(data_dir, 'filtered_indices.json')
        indices = read_json(inds_path)

    for filename in os.listdir(left_dir):
        if not filename.endswith('.png'):
            continue

        index = int(filename.split('.png')[0])

        if not index in indices:
            continue

        left_path = os.path.join(left_dir, filename)
        disparity_path = os.path.join(disparities_dir, filename.replace('.png', '.npy'))
        camera_info_path = os.path.join(camera_info_dir, filename.replace('.png', '.yml'))
        transform_path = os.path.join(transform_dir, filename.replace('.png', '.yml'))
        segmentation_path = os.path.join(segmentation_dir, filename.replace('.png', '.pkl'))
        clouds_camera_path = os.path.join(clouds_camera_dir, filename.replace('.png', '.npy'))
        clouds_world_path = os.path.join(clouds_world_dir, filename.replace('.png', '.npy'))

        if not os.path.exists(camera_info_path):
            raise RuntimeError('No camera_info_path for: ' + camera_info_path)
        
        if not os.path.exists(transform_path):
            raise RuntimeError('No transform_path for: ' + transform_path)
        
        left_paths.append(left_path)
        disparity_paths.append(disparity_path)
        camera_info_paths.append(camera_info_path)
        transform_paths.append(transform_path)
        segmentation_paths.append(segmentation_path)
        clouds_camera_paths.append(clouds_camera_path)
        clouds_world_paths.append(clouds_world_path)
        path_inds.append(index)

    to_ret = path_inds, left_paths, disparity_paths, camera_info_paths, transform_paths, segmentation_paths

    if include_cloud:
        to_ret = to_ret + (clouds_camera_paths, clouds_world_paths)

    if single:
        to_ret = list(to_ret)
        for i in range(len(to_ret)):
            to_ret[i] = to_ret[i][0] 
        to_ret = tuple(to_ret)
    return to_ret

    # if not single:
    #     return path_inds, left_paths, disparity_paths, camera_info_paths, transform_paths, segmentation_paths
    # else:
    #     return path_inds[0], left_paths[0], disparity_paths[0], camera_info_paths[0], transform_paths[0], segmentation_paths[0]

def get_discon_path(cloud_cam_path):
    return cloud_cam_path.replace('.npy', '_discon.png')

def get_retr(retr):
    if retr == 'LIST':
        return cv2.RETR_LIST
    
    if retr == "EXTERNAL":
        return cv2.RETR_EXTERNAL
    
    raise RuntimeError('Invalid retr: ' + retr)

def get_chain_approx(chain_approx):
    if chain_approx == 'SIMPLE':
        return cv2.CHAIN_APPROX_SIMPLE
    
    if chain_approx == "NONE":
        return cv2.CHAIN_APPROX_NONE
    
    raise RuntimeError('Invalid chain_approx: ' + chain_approx)

#create sphere cloud
def create_sphere_point_cloud(center, radius, num_points):
    points = []
    colors = []

    # Generate Fibonacci points
    golden_angle = np.pi * (3 - np.sqrt(5))
    offset = 2 / num_points
    for i in range(num_points):
        y = ((i * offset) - 1) + (offset / 2)
        radius_correction = np.sqrt(1 - np.power(y, 2))
        theta = (i % num_points) * golden_angle

        # Calculate point coordinates
        x = np.cos(theta) * radius_correction
        z = np.sin(theta) * radius_correction

        # Scale and translate the point
        point = np.array([x, y, z]) * radius + center
        points.append(point)
        colors.append([1.0, 0.0, 0.0])

    return np.vstack(points), np.vstack(colors)

#warp 2D points using homography
def warp_points(points, H):
    points_homo = np.ones((points.shape[0], 3))
    points_homo[:, 0:2] = points
    perspective_points_homo = (H @ points_homo.T).T
    perspective_points = perspective_points_homo[:, 0:2] / perspective_points_homo[:, 2:]

    return perspective_points



