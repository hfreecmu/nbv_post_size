import argparse
import os
import numpy as np

from nbv_utils import read_json, read_yaml, get_intrinsics

def get_depth_paths(data_dir):
    inds_path = os.path.join(data_dir, 'indices.json')
    indices = read_json(inds_path)

    camera_info_dir = os.path.join(data_dir, 'camera_info', 'right')
    depth_dir = os.path.join(data_dir, 'depth_images')
    left_image_dir = os.path.join(data_dir, 'rect_images', 'left')

    camera_info_paths = []
    depth_paths = []
    left_image_paths = []

    for filename in os.listdir(camera_info_dir):
        if not filename.endswith('.yml'):
            continue

        index = int(filename.split('.yml')[0])

        if not index in indices:
            continue

        camera_info_path = os.path.join(camera_info_dir, filename)
        depth_path = os.path.join(depth_dir, filename.replace('.yml', '.npy'))
        left_image_path = os.path.join(left_image_dir, filename.replace('.yml', '.png'))

        if not os.path.exists(depth_path):
            raise RuntimeError('No right image for: ' + depth_path)
        
        if not os.path.exists(left_image_path):
            raise RuntimeError('No left image for: ' + left_image_path)

        camera_info_paths.append(camera_info_path)
        depth_paths.append(depth_path)
        left_image_paths.append(left_image_path)

    return camera_info_paths, depth_paths, left_image_paths

def extract_disparity_from_depth(camera_info_path, depth_path):
    camera_info = read_yaml(camera_info_path)
    intrinsics = get_intrinsics(camera_info)

    depth = np.load(depth_path)

    baseline, f_norm, _, _ = intrinsics

    stub = depth / f_norm

    disparity = -baseline / stub

    disparity[np.isnan(disparity)] = -1
    
    return disparity

def extract_disparities_from_depth(data_dir):
    camera_info_paths, depth_paths, _ = get_depth_paths(data_dir)
    assert len(camera_info_paths) == len(depth_paths)

    disparities_dir = os.path.join(data_dir, 'disparities')
    if not os.path.exists(disparities_dir):
        os.mkdir(disparities_dir)

    for i in range(len(camera_info_paths)):
        camera_info_path = camera_info_paths[i]
        depth_path = depth_paths[i]

        disparity = extract_disparity_from_depth(camera_info_path, depth_path)

        basename = os.path.basename(depth_path).split('.npy')[0]
        disparity_output_path = os.path.join(disparities_dir, basename + '.npy')
        np.save(disparity_output_path, disparity)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    extract_disparities_from_depth(data_dir)