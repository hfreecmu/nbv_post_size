import argparse
import os
import numpy as np

from nbv_utils import get_paths, read_pickle, write_json
from nbv_utils import create_point_cloud, create_sphere_point_cloud

def get_base_fruiltet(data_dir, image_num, fruitlet_id, radius):
    path_ret = get_paths(data_dir, indices=[image_num], single=True, include_cloud=True)
    _, _, _, _, _, seg_path, _, cloud_world_path = path_ret
    
    world_points = np.load(cloud_world_path)
    
    segmentations = read_pickle(seg_path)
    seg_inds, _ = segmentations[fruitlet_id]

    seg_points = world_points[seg_inds[:, 0], seg_inds[:, 1]]

    good_seg_points = ~np.isnan(seg_points).any(axis=1)
    seg_points = seg_points[good_seg_points]
    
    centre = np.median(seg_points, axis=0)

    fruitlet_dict = {
        'image_num': image_num,
        'fruitlet_id': fruitlet_id,
        'radius': radius,
        'centre': [float(centre[0]), float(centre[1]), float(centre[2])],
    }

    output_path = os.path.join(data_dir, 'base_fruitlet.json')
    write_json(output_path, fruitlet_dict)

    sphere_points, sphere_colors = create_sphere_point_cloud(centre, radius, 1000)
    cloud_path = os.path.join(data_dir, 'target_centre.pcd')
    create_point_cloud(cloud_path, sphere_points, sphere_colors)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--image_num', type=int, required=True)
    parser.add_argument('--fruitlet_id', type=int, required=True)

    parser.add_argument('--radius', type=float, default=0.06)

    args = parser.parse_args()
    return args

#python3 6_select_base_fruitlet.py --data_dir $DATA_DIR --image_num $IMAGE_NUM --fruitlet_id $FRUITLET_ID
#not run in full pipeline. instead is run by extract_target_centres.py

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    image_num = args.image_num
    fruitlet_id = args.fruitlet_id
    radius = args.radius

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    get_base_fruiltet(data_dir, image_num, fruitlet_id, radius)