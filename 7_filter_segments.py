import argparse
import os
import cv2
import numpy as np
import distinctipy

from nbv_utils import get_paths, read_pickle, write_pickle
from nbv_utils import get_retr, get_chain_approx, read_json, write_json

def filter(left_path, seg_path, min_area, RETR, APPROX,
           world_path, sphere_centre, sphere_radius,
           radius_cutoff_pct):
    segmentations = read_pickle(seg_path)
    assert len(segmentations) < 255

    world_points = np.load(world_path)

    im = cv2.imread(left_path)
    full_im = im.copy()
    text_im = im.copy()
    filtered_segmentations = []

    seg_ids = np.zeros(im.shape[0:2], dtype=np.uint8) + 255
    scores = np.zeros(im.shape[0:2]) - 1

    for i in range(len(segmentations)):
        seg_inds, score = segmentations[i]

        #filter seg inds not outside of sphere
        #filter seg inds that have invalid disparity?
        #do we need to filter or just include all?
        #TODO do I want to do the latter?
        #decided we are not modifying fruitlet here
        seg_points = world_points[seg_inds[:, 0], seg_inds[:, 1]]
        non_nan_seg_points = seg_points[~np.isnan(seg_points).any(axis=1)]
        if non_nan_seg_points.shape[0] == 0:
            continue
        good_seg_points = np.linalg.norm((non_nan_seg_points - sphere_centre), axis=1) < sphere_radius
        #if over x% of points have been chopped because of sphere, continue
        good_rat = good_seg_points.sum() / non_nan_seg_points.shape[0]
        if good_rat < (1 - radius_cutoff_pct*0.01):
            continue 

        score_update_inds = np.where(scores[seg_inds[:, 0], seg_inds[:, 1]] < score)
        update_inds = (seg_inds[score_update_inds[0], 0], seg_inds[score_update_inds[0], 1])

        scores[update_inds] = score
        seg_ids[update_inds] = i

    filtered_ind = 0
    unique_ids = np.unique(seg_ids)
    colors = distinctipy.get_colors(unique_ids.shape[0])
    for un_id in range(unique_ids.shape[0]):
        color = ([int(255*colors[un_id][0]), int(255*colors[un_id][1]), int(255*colors[un_id][2])])

        i = unique_ids[un_id]

        if i == 255:
            continue

        seg_inds = np.argwhere(seg_ids == i)
        _, score = segmentations[i]

        if seg_inds.shape[0] < min_area:
            continue

        mask = np.zeros(im.shape[0:2], dtype=np.uint8)
        mask[seg_inds[:, 0], seg_inds[:, 1]] = 255
        #TODO why am I finding contours again?
        contours, _ = cv2.findContours(mask, RETR, APPROX)

        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            filtered_contours.append(contour)

        contour_im = np.zeros(im.shape[0:2], dtype=np.uint8)
        cv2.drawContours(contour_im, filtered_contours, -1, (255), -1)

        filtered_seg_inds = np.argwhere(contour_im > 0)
        if filtered_seg_inds.shape[0] == 0:
            continue
    
        full_im[filtered_seg_inds[:, 0], filtered_seg_inds[:, 1]] = color

        x0 = filtered_seg_inds[:, 1].min()
        x1 = filtered_seg_inds[:, 1].max() + 1
        y0 = filtered_seg_inds[:, 0].min()
        y1 = filtered_seg_inds[:, 0].max() + 1
        cent = (int((x0+x1)/2), int((y0+y1)/2))
        seg_string = 'Seg: ' + str(filtered_ind) + ', ' + "{:.2f}".format(score)
        text_im = cv2.putText(text_im, seg_string, cent, 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                         (0, 0, 0), 1)

        _, score = segmentations[i]
        filtered_segmentations.append((filtered_seg_inds, score))

        filtered_ind += 1

    has_fruitlet = len(filtered_segmentations) > 0
    return full_im, filtered_segmentations, text_im, has_fruitlet

def filter_segments(data_dir, min_area, retr, chain_approx, radius_cutoff_pct):
    path_ret = get_paths(data_dir, indices=None, include_cloud=True)
    left_paths = path_ret[1]
    seg_paths = path_ret[5]
    path_inds = path_ret[0]
    cloud_world_paths = path_ret[7]
    
    filtered_segment_dir = os.path.join(data_dir, 'segmentations_filtered')
    if not os.path.exists(filtered_segment_dir):
        os.mkdir(filtered_segment_dir)

    base_fruitlet_path = os.path.join(data_dir, 'base_fruitlet.json')
    base_fruilet = read_json(base_fruitlet_path)
    sphere_centre = np.array(base_fruilet['centre'])
    sphere_radius = base_fruilet['radius']

    new_path_inds = []

    for i in range(len(seg_paths)):
        left_path = left_paths[i]
        seg_path = seg_paths[i]
        path_ind = path_inds[i]
        cloud_world_path = cloud_world_paths[i]

        left_seg_im, filtered_segmentations, vis_im, has_fruitlet = filter(left_path, seg_path,
                                                                           min_area, retr, 
                                                                           chain_approx, cloud_world_path,
                                                                           sphere_centre, sphere_radius,
                                                                           radius_cutoff_pct)

        basename = os.path.basename(left_path).replace('.png', '.pkl')
        seg_output_path = os.path.join(filtered_segment_dir, basename)
        write_pickle(seg_output_path, filtered_segmentations)
        cv2.imwrite(seg_output_path.replace('.pkl', '.png'), left_seg_im)
        cv2.imwrite(seg_output_path.replace('.pkl', '_vis.png'), vis_im)

        if has_fruitlet:
            new_path_inds.append(path_ind)

    path_inds_output_path = os.path.join(data_dir, 'filtered_indices.json')
    write_json(path_inds_output_path, sorted(new_path_inds))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)

    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--retr', type=str, default='LIST')
    parser.add_argument('--chain_approx', type=str, default='SIMPLE')
    parser.add_argument('--radius_cutoff_pct', type=float, default=5.0)
    

    args = parser.parse_args()
    return args

#python3 7_filter_segments.py --data_dir $DATA_DIR

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    min_area = args.min_area
    retr = get_retr(args.retr)
    chain_approx = get_chain_approx(args.chain_approx)
    radius_cutoff_pct = args.radius_cutoff_pct

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    filter_segments(data_dir, min_area, retr, chain_approx, radius_cutoff_pct)