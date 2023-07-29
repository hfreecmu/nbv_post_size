import argparse
import os
import pandas as pd
import numpy as np

from extract_gt_labels import get_sub_dirs
from extract_target_centres import valid_bag_types
from skip_tags import skip_tags
from nbv_utils import read_json, write_json

def parse_date(date_string):
    year, month, day = date_string.split('-')
    year = int(year)
    month = int(month)
    day = int(day)

    return day, month, year

class Date:
    def __init__(self, date_str):
        day, month, year = parse_date(date_str)

        self.day = day
        self.month = month
        self.year = year

    def __str__(self):
        return '-'.join(["{:04d}".format(self.year), 
                         "{:02d}".format(self.month),
                         "{:02d}".format(self.day)])
    
    def __eq__(self, other):
        if not isinstance(other, Date):
            return False
        
        day_comp = (self.day == other.day)
        month_comp = (self.month == other.month)
        year_comp= (self.year == other.year)
        return (day_comp and month_comp and year_comp)

def filter_date_basenames(basenames, date, skip_tags):
    filtered_basenames = []
    for basename in basenames:
        if not str(date) in basename:
            continue

        if str(date) in skip_tags:
            tag_id = int(basename.split('_')[0])
            if tag_id in skip_tags[str(date)]:
                continue

        filtered_basenames.append(basename)
    
    return filtered_basenames

def insert_gt_data(gt_data, tag, fruitlet_num, gt_size=None, cv_size=None):
    if not tag in gt_data:
        raise RuntimeError('tag not in gt data')
    
    if not fruitlet_num in gt_data[tag]:
        gt_data[tag][fruitlet_num] = {
            'gt_size': -1,
            'cv_size': -1
        }

    if gt_size is not None:
        gt_data[tag][fruitlet_num]['gt_size'] = gt_size

    if cv_size is not None:
        if cv_size != -1:
            gt_data[tag][fruitlet_num]['cv_size'] = round(cv_size * 1000, 4)

def parse_gt_data(df, skip_tags_date):
    max_fruitlets = -1
    min_fruitlets = np.inf
    for key in df.keys():
        if not 'fl' in key:
            continue

        fruitlet_id = int(key.split('fl')[1])
        if fruitlet_id > max_fruitlets:
            max_fruitlets = fruitlet_id

        if fruitlet_id < min_fruitlets:
            min_fruitlets = fruitlet_id

    if max_fruitlets <= 0:
        raise RuntimeError('No fruitlets in spreadsheet')

    gt_data = {}
    for row_ind in range(df.shape[0]):
        tag = int(df['tag'][row_ind])

        if tag in skip_tags_date:
            continue

        assert tag not in gt_data

        gt_data[tag] = {}
        for fruitlet_id in range(min_fruitlets, max_fruitlets + 1):
            size = df['fl' + str(fruitlet_id)][row_ind]
            if np.isnan(size):
                continue

            insert_gt_data(gt_data, tag, fruitlet_id, gt_size=size)

    return gt_data

def parse_cv_data(data_dir, basenames, gt_data):
    num_spurious_dict = {}
    for basename in basenames:
        tag = int(basename.split('_')[0])

        subdir = os.path.join(data_dir, basename)
        label_ids_path = os.path.join(subdir, "gt_labels.json")
        if not os.path.exists(label_ids_path):
            raise RuntimeError('No label ids path for: ' + basename + 
                               '. Should it be in skip_tags?')
        
        clusters_path = os.path.join(subdir, 'associations', 'clusters.json')
        if not os.path.exists(clusters_path):
            raise RuntimeError('No clusters path for: ' + subdir)
        
        sizes_path = os.path.join(subdir, 'sizes', 'final_sizes.json')
        if not os.path.exists(sizes_path):
            raise RuntimeError('No sizes path for: ' + subdir)
        
        #key is fruitlet id from assoc
        #gt value will be gt label id
        label_ids = read_json(label_ids_path)["backward"]
        clusters= read_json(clusters_path)
        cv_sizes = read_json(sizes_path)
        
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
                insert_gt_data(gt_data, tag, gt_label_id, gt_size=None, cv_size=cv_size)

        num_spurious_dict[tag] = num_spurious


    new_data_dict = {
        'size_data': gt_data,
        'num_spurious': num_spurious_dict
    }

    return new_data_dict


def create_match_spreadsheet(data_dir, res_dir, bag_type, gt_csv, date_str, skip_tags):
    date = Date(date_str)
    basenames = get_sub_dirs(data_dir, bag_type)
    basenames = filter_date_basenames(basenames, date, skip_tags)

    gt_data = pd.read_excel(gt_csv, sheet_name=None)
    if not str(date) in gt_data.keys():
        raise RuntimeError('date not found in gt_data: ' + str(date))
    
    gt_data = gt_data[str(date)]

    if str(date) in skip_tags:
        skip_tags_date = skip_tags[str(date)]
    else:
        skip_tags_date = []

    gt_data = parse_gt_data(gt_data, skip_tags_date)

    if not (len(gt_data) == len(basenames)):
        #pass
        raise RuntimeError('gt and cv data num do not match')

    new_data_dict = parse_cv_data(data_dir, basenames, gt_data)

    output_dir = os.path.join(res_dir, '_'.join([str(date), bag_type]))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, 'sizing_results.json')
    write_json(output_path, new_data_dict, pretty=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--res_dir', required=True)
    parser.add_argument('--bag_type', required=True)
    parser.add_argument('--gt_csv', required=True)
    parser.add_argument('--date', default='2023-05-22')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    res_dir = args.res_dir
    bag_type = args.bag_type
    gt_csv = args.gt_csv
    date = args.date

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    if not os.path.exists(res_dir):
        raise RuntimeError('res_dir does not exist: ' + res_dir)
    
    if not bag_type in valid_bag_types:
        raise RuntimeError('Invalid bag_type: ' + bag_type + 
                           '. Choose one of ' + str(valid_bag_types) + '.')
    
    if not os.path.exists(gt_csv):
        raise RuntimeError('gt_csv does not exist: ' + gt_csv)
    
    create_match_spreadsheet(data_dir, res_dir, bag_type, gt_csv, date, skip_tags)