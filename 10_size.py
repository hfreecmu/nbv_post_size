import argparse
import os
import numpy as np
import cv2

from nbv_utils import read_json, read_pickle, get_paths, get_node_id
from nbv_utils import read_yaml, get_intrinsics, write_json

def size_fruitlet(im, disp_path, cam_info_path, seg_inds, 
         cloud_cam_path,
         ellipse,
         node_id):

    disparity = np.load(disp_path)
    camera_info = read_yaml(cam_info_path)
    intrinsics = get_intrinsics(camera_info)
    cam_points = np.load(cloud_cam_path)

    x_mid, y_mid = ellipse[0]
    width, height = ellipse[1]
    rot_ang = ellipse[2]
    rot_ang = np.deg2rad(rot_ang)

    if height < width:
        raise RuntimeError('height less than width for ellipse ' + node_id)
    
    x0 = x_mid - (width/2)*np.cos(rot_ang)
    x1 = x_mid + (width/2)*np.cos(rot_ang)

    y0 = y_mid - (width/2)*np.sin(rot_ang)
    y1 = y_mid + (width/2)*np.sin(rot_ang)

    #vis line
    cv2.line(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)

    #filter seg inds
    seg_points = cam_points[seg_inds[:, 0], seg_inds[:, 1]]
    good_seg_points = ~np.isnan(seg_points).any(axis=1)
    seg_inds = seg_inds[good_seg_points]

    disp_vals = disparity[seg_inds[:, 0], seg_inds[:, 1]]
    disp = np.median(disp_vals)
    #disp = np.max(disp_vals)

    baseline = -intrinsics[0]
    size = baseline * width / disp

    return size

def size_associate_max(size_dict, clusters):
    final_size_dict = {}
    for fruitlet_id in clusters['clusters']:
        max_size = None
        for node_id in clusters['clusters'][fruitlet_id]:
            if not node_id in size_dict:
                continue

            size = size_dict[node_id]['size']

            if (max_size is None) or (size > max_size):
                max_size = size

        if max_size is None:
            print('warning, could not size for fruitlet_id ' + str(fruitlet_id))
            final_size_dict[fruitlet_id] = -1
        else:
            final_size_dict[fruitlet_id] = max_size

    return final_size_dict

def size_associate(size_dict, clusters, bin_res):

    final_size_dict = {}
    for fruitlet_id in clusters['clusters']:
        min_occ_bin = None
        for node_id in clusters['clusters'][fruitlet_id]:
            if not node_id in size_dict:
                continue

            size = size_dict[node_id]['size']
            occ_rat = size_dict[node_id]['occ_rat']

            occ_bin = occ_rat // bin_res

            if (min_occ_bin is None) or (occ_bin < min_occ_bin):
                sizes = []
                min_occ_bin = occ_bin

            if (occ_bin == min_occ_bin):
                sizes.append(size)

        if min_occ_bin is None:
            print('warning, could not size for fruitlet_id ' + str(fruitlet_id))
            final_size_dict[fruitlet_id] = -1
        else:
            final_size_dict[fruitlet_id] = np.median(sizes)

    return final_size_dict


def size_fruitlets(data_dir, bin_res, use_max):
    clusters_path = os.path.join(data_dir, 'associations', 'clusters.json')
    if not os.path.exists(clusters_path):
        raise RuntimeError('clusters path does not exist: ' + clusters_path)
    
    ellipses_path = os.path.join(data_dir, 'ellipses', 'ellipses.pkl')
    if not os.path.exists(ellipses_path):
        raise RuntimeError('ellipses path does not exist: ' + ellipses_path)
    
    size_dir = os.path.join(data_dir, 'sizes')
    if not os.path.exists(size_dir):
        os.mkdir(size_dir)
     
    clusters = read_json(clusters_path)
    ellipses_dict = read_pickle(ellipses_path)

    path_ret = get_paths(data_dir, indices=None, use_filter_segs=True, include_cloud=True, use_filter_indices=True)
    image_inds, left_paths, disp_paths, cam_info_paths, _, seg_paths, cloud_cam_paths, _ = path_ret

    size_dict = {}
    for i in range(len(image_inds)):
        image_ind = image_inds[i]
        left_path = left_paths[i]
        seg_path = seg_paths[i]
        segmentations = read_pickle(seg_path)

        im = cv2.imread(left_path)

        for seg_id in range(len(segmentations)):
            node_id = get_node_id(image_ind, seg_id)
            seg_inds, _ = segmentations[seg_id]
            if not node_id in ellipses_dict:
                continue

            ellipse = ellipses_dict[node_id]['ellipse']
            occ_rat = ellipses_dict[node_id]["occ_rat"]
            size = size_fruitlet(im, disp_paths[i], cam_info_paths[i],
                                 seg_inds, cloud_cam_paths[i],
                                 ellipse, node_id) 
            size_dict[node_id] = {'size': size, 
                                  'occ_rat': occ_rat}

        im_path = os.path.join(size_dir, str(image_ind) + '_line_im.png')
        cv2.imwrite(im_path, im)

    size_path = os.path.join(size_dir, 'full_sizes.json')
    write_json(size_path, size_dict, pretty=True)

    if not use_max:
        final_size_dict = size_associate(size_dict, clusters, bin_res)
    else:
        final_size_dict = size_associate_max(size_dict, clusters)

    final_size_path = os.path.join(size_dir, 'final_sizes.json')
    write_json(final_size_path, final_size_dict, pretty=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--bin_res', type=int, default=5)
    parser.add_argument('--use_max', action='store_true')
    
    args = parser.parse_args()
    return args

#python3 10_size.py --data_dir $DATA_DIR

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    bin_res = args.bin_res
    use_max = args.use_max

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    size_fruitlets(data_dir, bin_res, use_max)