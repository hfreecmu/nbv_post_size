import os
import argparse
import numpy as np
import matplotlib
import cv2
from sklearn.cluster import DBSCAN

from nbv_utils import read_json, write_pickle

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

#bg_color = np.array([0.64, 0.86, 0.91])
#bg_hsv = matplotlib.colors.rgb_to_hsv(bg_color.astype(float))

def get_image_cloud_paths(data_dir):
    inds_path = os.path.join(data_dir, 'indices.json')
    indices = read_json(inds_path)

    left_dir = os.path.join(data_dir, 'rect_images', 'left')
    clouds_camera_dir = os.path.join(data_dir, 'clouds_camera')

    left_paths = []
    cloud_paths = []

    for filename in os.listdir(left_dir):
        if not filename.endswith('.png'):
            continue

        index = int(filename.split('.png')[0])

        if not index in indices:
            continue

        left_path = os.path.join(left_dir, filename)
        left_paths.append(left_path)

        cloud_path = os.path.join(clouds_camera_dir, filename.replace('.png', '.npy'))
        cloud_paths.append(cloud_path)

    return left_paths, cloud_paths

def segment_image_hsv(image_path, pointcloud_path, hsv_colors=hsv_colors, 
                      hsv_thresh=0.15, min_area=100, db_eps=0.01, db_min_samples=20):
                      #bg_thresh=0.15):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    n_rows, n_cols, _ = im.shape

    cam_points = np.load(pointcloud_path)

    hsv_image = matplotlib.colors.rgb_to_hsv(im.astype(float)/255)
    hsv_image_reshape = hsv_image.reshape(-1, 3)

    #to ensure not order dependant
    hsv_inds = np.random.permutation(hsv_colors.shape[0])
    hsv_colors = hsv_colors[hsv_inds]

    dists = np.linalg.norm((hsv_image_reshape - hsv_colors[:, None]), axis=2) #hsv_colors.shape[0] x hsv_image_reshape.shape[0]
    min_inds = np.argmin(dists, axis=0) #hsv_image_reshape.shape[0]
    min_dists = np.choose(min_inds, dists) #hsv_image_reshape.shape[0]
    seg_inds = np.argwhere(min_dists < hsv_thresh)[:, 0] #hsv_image_reshape.shape[0]

    #now do bg segmentation
    #if uncomment this, make sure to remove bg inds in code as it's not used
    #dists = np.linalg.norm(hsv_image_reshape - bg_hsv, axis=1) #hsv_image_reshape.shape[0]
    #bg_inds = np.argwhere(dists < bg_thresh)[:, 0]

    #TODO not sure

    seg_ids = np.zeros((hsv_image_reshape.shape[0]), dtype=np.uint8) + 255
    seg_ids[seg_inds] = min_inds[seg_inds]

    seg_ids = seg_ids.reshape(n_rows, n_cols)

    #TODO not sure if min thresh will help
    unique_ids = np.unique(seg_ids)
    clustered_seg_ids = np.zeros(seg_ids.shape, dtype=np.uint8) - 1
    curr_id = 0
    
    for id in unique_ids:
        if id == 255:
            continue

        seg_inds = np.argwhere(seg_ids == id)
        
        if seg_inds.shape[0] > 20000:
            print('ISSUE SEG HAPPENED HERE. SKIPPING BUT MAY NEED TO RETUNE.')
            continue
        
        seg_points = cam_points[seg_inds[:, 0], seg_inds[:, 1]]

        non_nan_inds = np.argwhere(~np.isnan(seg_points).any(axis=1))[:, 0]
        non_nan_seg_points = seg_points[non_nan_inds]

        if non_nan_seg_points.shape[0] == 0:
            continue

        db = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(non_nan_seg_points)
        labels = db.labels_

        for label_id in np.unique(labels):
            if label_id == -1:
                continue

            label_inds = np.argwhere(labels == label_id)[:, 0]
            non_nan_label_inds = non_nan_inds[label_inds]
            seg_label_inds = seg_inds[non_nan_label_inds]

            if seg_label_inds.shape[0] < min_area:
                continue

            clustered_seg_ids[seg_label_inds[:, 0], seg_label_inds[:, 1]] = curr_id
            curr_id += 1

    seg_im = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    segmentations = []
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for id in np.unique(clustered_seg_ids):
        if id == 255:
            continue

        ind = len(segmentations)

        seg_inds = np.argwhere(clustered_seg_ids == id)
        seg_im[seg_inds[:, 0], seg_inds[:, 1]] = 255
        segmentations.append((seg_inds, 0.95))

        x0 = np.min(seg_inds[:, 1])
        x1 = np.max(seg_inds[:, 1])
        y0 = np.min(seg_inds[:, 0])
        y1 = np.max(seg_inds[:, 0])

        cent = (int((x0+x1)/2), int((y0+y1)/2))
        seg_string = 'Seg: ' + str(ind) + ', ' + "{:.2f}".format(0.95)
        im = cv2.putText(im, seg_string, cent, 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                         (0, 0, 0), 1)
        
    return seg_im, segmentations, im


def segment_hsv(data_dir, hsv_colors):
    left_paths, cloud_paths = get_image_cloud_paths(data_dir)

    segment_dir = os.path.join(data_dir, 'segmentations')
    if not os.path.exists(segment_dir):
        os.mkdir(segment_dir)

    for i in range(len(left_paths)):
        left_path = left_paths[i]
        cloud_path = cloud_paths[i]

        left_seg_im, segmentations, vis_im = segment_image_hsv(left_path, cloud_path, hsv_colors)

        basename = os.path.basename(left_path).replace('.png', '.pkl')
        left_seg_output_path = os.path.join(segment_dir, basename)
        write_pickle(left_seg_output_path, segmentations)
        cv2.imwrite(left_seg_output_path.replace('.pkl', '.png'), left_seg_im)
        cv2.imwrite(left_seg_output_path.replace('.pkl', '_vis.png'), vis_im)

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
    
    segment_hsv(data_dir, hsv_colors)