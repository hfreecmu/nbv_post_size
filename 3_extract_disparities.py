import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/RAFT-Stereo')
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/RAFT-Stereo/core')

import argparse
import os
import json
import numpy as np
import torch
from PIL import Image

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

DEVICE = 'cuda'

def read_json(path):
    with open(path, 'r') as f:
        json_to_read = json.loads(f.read())
    
    return json_to_read

class Dummy:
    def __init__(self) -> None:
        pass

def get_irvc_model_args(restore_ckpt):
    args = Dummy()
    args.restore_ckpt = restore_ckpt
    args.context_norm = 'instance'

    args.shared_backbone = False
    args.n_downsample = 2
    args.n_gru_layers = 3
    args.slow_fast_gru = False
    args.valid_iters = 32
    args.corr_implementation = 'reg'
    args.mixed_precision = False
    args.hidden_dims = [128]*3
    args.corr_levels = 4
    args.corr_radius = 4

    return args  

def load_raft_model(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model

def load_raft_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def extract_disparity(model, left_im_path, right_im_path, valid_iters):
    image1 = load_raft_image(left_im_path)
    image2 = load_raft_image(right_im_path)

    with torch.no_grad():
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = model(image1, image2, iters=valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()

    disparity = -flow_up.cpu().numpy().squeeze()
    return disparity

def get_image_paths(data_dir):
    inds_path = os.path.join(data_dir, 'indices.json')
    indices = read_json(inds_path)

    left_dir = os.path.join(data_dir, 'rect_images', 'left')
    right_dir = os.path.join(data_dir, 'rect_images', 'right')

    left_paths = []
    right_paths = []

    for filename in os.listdir(left_dir):
        if not filename.endswith('.png'):
            continue

        index = int(filename.split('.png')[0])

        if not index in indices:
            continue

        left_path = os.path.join(left_dir, filename)
        right_path = os.path.join(right_dir, filename)

        if not os.path.exists(right_path):
            raise RuntimeError('No right image for: ' + right_path)
        
        left_paths.append(left_path)
        right_paths.append(right_path)

    return left_paths, right_paths

def extract_disparities(data_dir, raft_restore_ckpt):
    left_paths, right_paths = get_image_paths(data_dir)
    assert len(left_paths) == len(right_paths)

    disparities_dir = os.path.join(data_dir, 'disparities')
    if not os.path.exists(disparities_dir):
        os.mkdir(disparities_dir)

    raft_args = get_irvc_model_args(raft_restore_ckpt)
    raft_model = load_raft_model(raft_args)

    for i in range(len(left_paths)):
        left_path = left_paths[i]
        right_path = right_paths[i]

        disparity = extract_disparity(raft_model, left_path, right_path, raft_args.valid_iters)
        
        basename = os.path.basename(left_path).split('.png')[0]
        disparity_output_path = os.path.join(disparities_dir, basename + '.npy')
        np.save(disparity_output_path, disparity) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--raft_restore_ckpt', required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    raft_restore_ckpt = args.raft_restore_ckpt

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    if not os.path.exists(raft_restore_ckpt):
        raise RuntimeError('raft_restore_ckpt does not exist: ' + raft_restore_ckpt)
    
    extract_disparities(data_dir, raft_restore_ckpt)