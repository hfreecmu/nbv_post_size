import argparse
import os
import numpy as np

from annotator import Annotate
from extract_target_centres import valid_bag_types

def get_sub_dirs(data_dir, bag_type):
    basenames = []
    tag_ids = []
    for subdir_name in os.listdir(data_dir):
        if not bag_type in subdir_name:
            continue

        subdir = os.path.join(data_dir, subdir_name)
        if not os.path.isdir(subdir):
            continue
    
        clusters_path = os.path.join(subdir, 'associations', 'clusters.json')
        if not os.path.exists(clusters_path):
            #raise RuntimeError('No clusters path for: ' + subdir)
            print('No clusters for: ', subdir_name)
            continue

        tag_id = int(subdir_name.split('_')[0])
        tag_ids.append(tag_id)
        basenames.append(subdir_name)

    sorted_inds = np.argsort(tag_ids)
    basenames = [basenames[i] for i in sorted_inds]

    return basenames

def extract_gt_labels_full(data_dir, bag_type, start_dir):
    basenames = get_sub_dirs(data_dir, bag_type)

    if start_dir is not None:
        if not start_dir in basenames:
            raise RuntimeError('start_dir not in subdirs: ' + start_dir)
        dir_index = basenames.index(start_dir)
    else:
        dir_index = 0
    
    should_quit = False
    while not should_quit:
        sub_dir = os.path.join(data_dir, basenames[dir_index])
        annotator = Annotate(sub_dir)
        quit_res = annotator.annotate()
        if quit_res == 'hard':
            should_quit = True
        elif quit_res == 'left':
            if dir_index > 0:
                dir_index -= 1
        elif quit_res == 'right':
            if dir_index < len(basenames) - 1:
                dir_index += 1
        else:
            raise RuntimeError('Illegal annotate res: ', quit_res)

def check_no_gt_sizes(data_dir, bag_type):
    basenames = get_sub_dirs(data_dir, bag_type)
    no_gt_labels = []

    for basename in basenames:
        gt_label_path = os.path.join(data_dir, basename, "gt_labels.json")
        if not os.path.exists(gt_label_path):
            no_gt_labels.append(basename)

    return no_gt_labels
    

#DATA_DIR=DATA_DIR
#BAG_TYPE=TYPE_OF_BAG
#python3 extract_gt_labels.py --data_dir $DATA_DIR --bag_type $BAG_TYPE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--bag_type', required=True)
    
    parser.add_argument('--start_dir', default=None)
    parser.add_argument('--check_only', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    bag_type = args.bag_type
    start_dir = args.start_dir
    check_only = args.check_only

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    if not bag_type in valid_bag_types:
        raise RuntimeError('Invalid bag_type: ' + bag_type + 
                           '. Choose one of ' + str(valid_bag_types) + '.')
    
    if not check_only:
        extract_gt_labels_full(data_dir, bag_type, start_dir)
    else:
        no_gt_labels = check_no_gt_sizes(data_dir, bag_type)
        print('No gt labels for: ')
        for basename in no_gt_labels:
            print(basename)