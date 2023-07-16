import argparse
import os
import numpy as np
import json

def read_file(path):
    with open(path) as f:
        lines = f.readlines()

    vals = []
    for line in lines:
        vals.append(int(line))

    return vals

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def process(data_dir):
    timestamps_path = os.path.join(data_dir, 'timestamps.txt')
    start_capture_path = os.path.join(data_dir, 'start_captures.txt')

    timestamps = read_file(timestamps_path)
    start_captures = read_file(start_capture_path)

    if len(start_captures) == 0:
        print('WARNING: No start captures')
        capture_time = -1
    else:
        capture_time = np.max(start_captures)

    timestamps = np.array(timestamps)

    inds = np.argwhere(timestamps >= capture_time)

    #tag is in first image so exclude it
    inds = inds[1:]

    inds = inds[:, 0].tolist()

    output_path = os.path.join(data_dir, 'indices.json')
    write_json(output_path, inds)

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
    
    process(data_dir)
    
