import os
import argparse
import cv2
import numpy as np
import matplotlib

from extract_target_centres import valid_bag_types
from parse_sdf_world import parse_world
from nbv_utils import read_json, parse_node_id
from nbv_utils import read_pickle, write_json
import scipy.optimize

purple = np.array([255, 100, 255])
blue = np.array([40, 40, 255])
orange = np.array([255, 140, 67])
red = np.array([253, 39, 39])
yellow = np.array([255, 255, 41])
teal = np.array([42, 254, 255])

purple_hsv = matplotlib.colors.rgb_to_hsv(purple.astype(float)/255)
blue_hsv = matplotlib.colors.rgb_to_hsv(blue.astype(float)/255)
orange_hsv = matplotlib.colors.rgb_to_hsv(orange.astype(float)/255)
red_hsv = matplotlib.colors.rgb_to_hsv(red.astype(float)/255)
yellow_hsv = matplotlib.colors.rgb_to_hsv(yellow.astype(float)/255)
teal_hsv = matplotlib.colors.rgb_to_hsv(teal.astype(float)/255)

hsv_colors = np.vstack((purple_hsv, blue_hsv, orange_hsv, red_hsv, yellow_hsv, teal_hsv))

def parse_exp_dirname(basename):
    cluster_tex, cluster_num, w_0, w_1 = basename.split('_')
    cluster_id = '_'.join([cluster_tex, cluster_num])
    world_id = '_'.join([w_0, w_1])

    return cluster_id, world_id

def get_sub_dirs(data_dir, bag_type):
    basenames = []
    for subdir_name in os.listdir(data_dir):
        if not bag_type in subdir_name:
            continue

        subdir = os.path.join(data_dir, subdir_name)
        if not os.path.isdir(subdir):
            continue
    
        clusters_path = os.path.join(subdir, 'associations', 'clusters.json')
        if not os.path.exists(clusters_path):
            raise RuntimeError('No clusters path for: ' + subdir)
            #print('No clusters for: ', subdir_name)
            #continue

        basenames.append(subdir_name)

    return basenames

def get_clusters(data_dir):
    clusters_path = os.path.join(data_dir, 'associations', 'clusters.json')
    if not os.path.exists(clusters_path):
        raise RuntimeError('clusters path does not exist: ' + clusters_path)

    clusters = read_json(clusters_path)
    return clusters

def get_cluster_colors(data_dir):
    clusters = get_clusters(data_dir)
    
    color_vals = []
    for fruitlet_id in clusters["clusters"]:
        med_vals = None
        for node_id in clusters["clusters"][fruitlet_id]:
            image_ind, seg_id = parse_node_id(node_id)
            seg_path = os.path.join(data_dir, 'segmentations_filtered', "{:06d}.pkl".format(image_ind))
            segmentations = read_pickle(seg_path)
            seg_inds, _ = segmentations[seg_id]

            im_path = os.path.join(data_dir, 'rect_images', 'left', 
                                   os.path.basename(seg_path).replace('.pkl', '.png'))
            im = cv2.imread(im_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = matplotlib.colors.rgb_to_hsv(im.astype(float)/255)
            
            if med_vals is None:
                med_vals = im[seg_inds[:, 0], seg_inds[:, 1]]
            else:
                med_vals = np.concatenate((med_vals, im[seg_inds[:, 0], seg_inds[:, 1]]))

        med_val = np.median(med_vals, axis=0)
        color_vals.append([int(fruitlet_id), med_val])

    return color_vals

def insert_gt_data(gt_data, cluster_id, fruitlet_num, gt_size=None, cv_size=None):
    if not cluster_id in gt_data:
        raise RuntimeError('cluster_id in gt data')
    
    if not fruitlet_num in gt_data[cluster_id]:
        gt_data[cluster_id][fruitlet_num] = {
            'gt_size': -1,
            'cv_size': -1
        }

    if gt_size is not None:
        gt_data[cluster_id][fruitlet_num]['gt_size'] = round(gt_size * 1000, 4)

    if cv_size is not None:
        if cv_size != -1:
            gt_data[cluster_id][fruitlet_num]['cv_size'] = round(cv_size * 1000, 4)


def extract_gt_label(input_dir, output_dir, basename, model_dir, gt_data):
    _, world_id = parse_exp_dirname(basename)
    
    world_path = os.path.join(input_dir, basename, world_id + '.world')
    fruitlet_dict = parse_world(world_path, model_dir)
    fruitlet_dict_keys = list(fruitlet_dict.keys())

    subdir = os.path.join(output_dir, basename)

    color_vals = get_cluster_colors(subdir)

    C = np.zeros((len(fruitlet_dict_keys), len(color_vals)))
    for i in range(len(fruitlet_dict_keys)):
        key = fruitlet_dict_keys[i]
        gt_fruitlet_id = int(key.split('_')[2])
        gt_color = hsv_colors[gt_fruitlet_id]

        for j in range(len(color_vals)):
            _, cv_color = color_vals[j]
            C[i, j] = np.linalg.norm(gt_color - cv_color)

        #assume all size scales are same
        size = fruitlet_dict[key]['scale'][0]
        insert_gt_data(gt_data, basename, i, gt_size=size, cv_size=None)

    
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)
    annotation_dict = {"forward": {},
                       "backward": {}}
    
    for ind in range(len(row_ind)):
        i = row_ind[ind]
        j = col_ind[ind]
        
        key = fruitlet_dict_keys[i]
        gt_fruitlet_id = int(key.split('_')[2])

        cv_fruitlet_id, _ = color_vals[j]

        annotation_dict['forward'][gt_fruitlet_id] = cv_fruitlet_id
        annotation_dict['backward'][cv_fruitlet_id] = gt_fruitlet_id

    annotation_path = os.path.join(subdir, 'gt_labels.json')
    write_json(annotation_path, annotation_dict)
    

def extract_gt_labels_full(input_dir, output_dir, bag_type, model_dir, res_dir):
    basenames = get_sub_dirs(output_dir, bag_type)
    gt_data = {}
    num_spurious_dict = {}
    for basename in basenames:
        gt_data[basename] = {}
        extract_gt_label(input_dir, output_dir, basename, model_dir, gt_data)

        subdir = os.path.join(output_dir, basename)
        sizes_path = os.path.join(subdir, 'sizes', 'final_sizes.json')
        clusters_path = os.path.join(subdir, 'associations', 'clusters.json')
        label_ids_path = os.path.join(subdir, "gt_labels.json")

        clusters= read_json(clusters_path)
        cv_sizes = read_json(sizes_path)
        label_ids = read_json(label_ids_path)["backward"]

        num_spurious = 0
        for key in clusters["clusters"]:
            fruitlet_id = int(key)
            if not (str(fruitlet_id) in cv_sizes):
                raise RuntimeError(str(fruitlet_id) + ' not in cv_sizes for ' 
                                   + subdir)
            
            cv_size = cv_sizes[str(fruitlet_id)]
            if not str(fruitlet_id) in label_ids:
                num_spurious += 1
            else:
                gt_label_id = int(label_ids[str(fruitlet_id)])
                insert_gt_data(gt_data, basename, gt_label_id, gt_size=None, cv_size=cv_size)

        num_spurious_dict[basename] = num_spurious
    
    new_data_dict = {
        'size_data': gt_data,
        'num_spurious': num_spurious_dict
    }


    output_path = os.path.join(res_dir, 'sizing_results.json')
    write_json(output_path, new_data_dict, pretty=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bag_type', required=True)
    parser.add_argument('--model_dir', required=True)
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    bag_type = args.bag_type
    model_dir = args.model_dir
    res_dir = args.res_dir

    if not os.path.exists(input_dir):
        raise RuntimeError('input_dir does not exist: ' + input_dir)
    
    if not os.path.exists(output_dir):
        raise RuntimeError('input_dir does not exist: ' + output_dir)
    
    if not bag_type in valid_bag_types:
        raise RuntimeError('Invalid bag_type: ' + bag_type + 
                           '. Choose one of ' + str(valid_bag_types) + '.')
    
    if not os.path.exists(model_dir):
        raise RuntimeError('model_dir does not exist: ' + model_dir)
    
    if not os.path.exists(res_dir):
        raise RuntimeError('res_dir does not exist: ' + res_dir)
    
    extract_gt_labels_full(input_dir, output_dir, bag_type, model_dir, res_dir)
