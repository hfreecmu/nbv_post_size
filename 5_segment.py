import argparse
import os
import json
import torch
import cv2
import numpy as np
import pickle

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

def read_json(path):
    with open(path, 'r') as f:
        json_to_read = json.loads(f.read())
    
    return json_to_read

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_seg_model(model_file, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_file 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_file)['model'])
    model.eval()

    return model

def segment_image(model, image_path):
    im = cv2.imread(image_path)
    seg_im = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    segmentations = []

    image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image}
    with torch.no_grad():
        outputs = model([inputs])[0]

    masks = outputs['instances'].get('pred_masks').to('cpu').numpy()
    boxes = outputs['instances'].get('pred_boxes').to('cpu')
    scores = outputs['instances'].get('scores').to('cpu').numpy()

    num = len(boxes)
    #for segmentation
    assert num < 254

    for i in range(num):        
        seg_inds = np.argwhere(masks[i, :, :] > 0)
        seg_im[seg_inds[:, 0], seg_inds[:, 1]] = 255
        segmentations.append((seg_inds, scores[i]))

        x0, y0, x1, y1 = boxes[i].tensor.numpy()[0]
        cent = (int((x0+x1)/2), int((y0+y1)/2))
        seg_string = 'Seg: ' + str(i) + ', ' + "{:.2f}".format(scores[i])
        im = cv2.putText(im, seg_string, cent, 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                         (0, 0, 0), 1)

    return seg_im, segmentations, im

def get_image_paths(data_dir):
    inds_path = os.path.join(data_dir, 'indices.json')
    indices = read_json(inds_path)

    left_dir = os.path.join(data_dir, 'rect_images', 'left')

    left_paths = []

    for filename in os.listdir(left_dir):
        if not filename.endswith('.png'):
            continue

        index = int(filename.split('.png')[0])

        if not index in indices:
            continue

        left_path = os.path.join(left_dir, filename)
        left_paths.append(left_path)

    return left_paths

def segment(data_dir, model_path, score_thresh):
    left_paths = get_image_paths(data_dir)

    segment_dir = os.path.join(data_dir, 'segmentations')
    if not os.path.exists(segment_dir):
        os.mkdir(segment_dir)

    seg_model = load_seg_model(model_path, score_thresh)

    for left_path in left_paths:
        left_seg_im, segmentations, vis_im = segment_image(seg_model, left_path)

        basename = os.path.basename(left_path).replace('.png', '.pkl')
        left_seg_output_path = os.path.join(segment_dir, basename)
        write_pickle(left_seg_output_path, segmentations)
        cv2.imwrite(left_seg_output_path.replace('.pkl', '.png'), left_seg_im)
        cv2.imwrite(left_seg_output_path.replace('.pkl', '_vis.png'), vis_im)

#SEG_MODEL_PATH='/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/turk/mask_rcnn/mask_best.pth'
#python3 5_segment.py --data_dir $DATA_DIR --model_path $SEG_MODEL_PATH

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--score_thresh', type=float, default=0.4)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    model_path = args.model_path
    score_thresh = args.score_thresh

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    if not os.path.exists(model_path):
        raise RuntimeError('model_path does not exist: ' + model_path)
    
    segment(data_dir, model_path, score_thresh)

