import argparse
import os

from annotator import Annotate

def gt_label(data_dir):

    annotator = Annotate(data_dir)
    _ = annotator.annotate()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    
    args = parser.parse_args()
    return args

#python3 11_gt_size.py --data_dir $DATA_DIR

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    gt_label(data_dir)
    