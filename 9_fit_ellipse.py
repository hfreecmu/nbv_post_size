import argparse
import os
import cv2
import numpy as np

from nbv_utils import get_paths, read_json, read_pickle, get_node_id
from nbv_utils import get_discon_path, read_yaml, get_intrinsics
from nbv_utils import get_retr, get_chain_approx, compute_points
from nbv_utils import write_pickle

def dot_prod(A, B):
    return (A*B).sum(axis=1)

# def correct_discon_points(points, discon_map, surface_norm, MAG):
#     for i in range(points.shape[0]):
#         x0, y0 = points[i]
#         dx, dy = surface_norm[i]

#         for d_ind in range(1, 1000):
#             y0_use = int(np.round(y0))
#             x0_use = int(np.round(x0))
#             discons = discon_map[y0_use-MAG:y0_use+MAG+1, x0_use-MAG:x0_use+MAG+1] / 255
#             is_discon = discons.mean() > 0.5

#             if not is_discon:
#                 break

#             y0 = y0 - d_ind*dy
#             x0 = x0 - d_ind*dx

#         if is_discon:
#             raise RuntimeError('Still discon safety guard')

#         points[i] = [x0, y0]

#TODO must be better way to do this 
# but have to put it to rest as tired
def get_occluded_points(points, z_points, surface_norm, 
                        discon_map, MAG, 
                        discon_thresh=0.5,
                        z_thresh = 0.0000):
    is_occluded = []
    for i in range(points.shape[0]):
        x0, y0 = points[i]
        dx, dy = surface_norm[i]
        is_discon = (discon_map[y0, x0] > 0)  

        x1 = int(np.round(x0 + dx*MAG))
        y1 = int(np.round(y0 + dy*MAG))
        discons = discon_map[y1-MAG:y1+MAG+1, x1-MAG:x1+MAG+1] / 255
        

        #if not a discontinuity it is not an occlusion as same object
        discons = np.mean(discons)
        if discons < discon_thresh:
            is_occluded.append(False)
            continue

        # if it is a discon, need to determine if in front or behind
        x2 = int(np.round(x0 - dx))
        y2 = int(np.round(y0 - dy))
        
        z0s = z_points[y2-MAG:y2+MAG+1, x2-MAG:x2+MAG+1]
        z1s = z_points[y1-MAG:y1+MAG+1, x1-MAG:x1+MAG+1]

        z0 = np.median(z0s)
        z1 = np.median(z1s)
        
        if z0 < z1:
            is_occluded.append(False)
            continue

        is_occluded.append(True)

    return np.array(is_occluded)


def fit_ellipse_image(clusters, ellipse_dict, output_dir,
                      image_ind, image_path, 
                      disp_path, seg_path, cam_info_path, 
                      cloud_cam_path,
                      RETR, APPROX, 
                      MAG, occ_thresh):
    
    discon_path = get_discon_path(cloud_cam_path)

    im = cv2.imread(image_path)
    segmentations = read_pickle(seg_path)
    cam_points = np.load(cloud_cam_path)
    disparity = np.load(disp_path)
    discon_map = cv2.imread(discon_path, cv2.IMREAD_GRAYSCALE)
    camera_info = read_yaml(cam_info_path)

    intrinsics = get_intrinsics(camera_info)
    z_points = compute_points(disparity, intrinsics)[:, :, 2]

    #not sure if I want to do this
    bad_points = np.isnan(cam_points).any(axis=2)
    im[bad_points] = [0, 0, 0]

    occ_im = im.copy()
    ellipse_im = im.copy()

    for seg_id in range(len(segmentations)):
        node_id = get_node_id(image_ind, seg_id)
        if not node_id in clusters['fruitlets']:
            continue

        if node_id in ellipse_dict:
            raise RuntimeError('node_id in ellipse dict already: ' + node_id)

        seg_inds, _ = segmentations[seg_id]
        #when we fit contours we want the original image
        #just cause discontinuities can break it apart
        seg_inds_orig = seg_inds.copy()

        #remove discon and far away points from seg inds
        seg_points = cam_points[seg_inds[:, 0], seg_inds[:, 1]]
        good_seg_points = ~np.isnan(seg_points).any(axis=1)
        seg_inds = seg_inds[good_seg_points]

        #find contours from ALL orig seg points
        mask = np.zeros(im.shape[0:2], dtype=np.uint8)
        mask[seg_inds_orig[:, 0], seg_inds_orig[:, 1]] = 255
        contours, _ = cv2.findContours(mask, RETR, APPROX)

        if len(contours) > 1:
            print('Too many contours for ' + node_id + '. Skipping')
            continue

        #going to assume they are in a circle
        #we DO NOT filter contour points as that effected
        #surface normals and what not
        #plus ellipses looked better
        #if we want to, look at the debug script
        contour = contours[0]
        points = np.array(contour)
        points = points[:, 0, :]

        num_points = points.shape[0]
        n_inds = np.arange(num_points)
        neg_inds = (n_inds - 1) % num_points
        pos_inds = (n_inds + 1) % num_points

        norm_prep_vecs = (points[pos_inds] - points[neg_inds])
        norm_norms = np.linalg.norm(norm_prep_vecs, axis=1)

        #filter out norms that are 0. This can happen if there's 
        #a single pixel in one direction
        bad_norm_vals = (norm_norms == 0)
        norm_prep_vecs = norm_prep_vecs[~bad_norm_vals]
        norm_norms = norm_norms[~bad_norm_vals]
        points = points[~bad_norm_vals]

        norm_prep_vecs = norm_prep_vecs / np.linalg.norm(norm_prep_vecs, axis=1).reshape((-1, 1))
        surface_norm = np.zeros_like(norm_prep_vecs)
        surface_norm[:, 0] = norm_prep_vecs[:, 1]
        surface_norm[:, 1] = -norm_prep_vecs[:, 0]

        centroid = np.mean(points, axis=0)
        centroid_vec = centroid - points
        centroid_vec = centroid_vec / np.linalg.norm(centroid_vec, axis=1).reshape((-1, 1))

        #want to point opposite direction or centroid
        neg_dot_inds = dot_prod(surface_norm, centroid_vec) > 0
        surface_norm[neg_dot_inds] = -surface_norm[neg_dot_inds]

        is_occluded = get_occluded_points(points, z_points, surface_norm, discon_map, MAG)

        #vis occluded points
        for i in range(points.shape[0]):
            x0, y0 = points[i]
            if is_occluded[i]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.circle(occ_im, (x0,y0), radius=0, color=color, thickness=-1)

        #don't ellipse fit if too many points out
        occ_rat = 1 - np.sum(is_occluded) / is_occluded.shape[0]

        if occ_rat < occ_thresh:
            continue

        #fit the ellipse
        ellipse_points = points[~is_occluded]
        ellipse_points = np.expand_dims(ellipse_points, axis=1)
        ellipse = cv2.fitEllipse(ellipse_points)

        #vis ellipse
        cv2.ellipse(ellipse_im, ellipse, (255, 0, 0), 2)

        #add to ellipse dict
        ellipse_dict[node_id] = {"ellipse": ellipse,
                                 "occ_rat": 1 - occ_rat}

    occ_im_path= os.path.join(output_dir, str(image_ind) + '_occ_im.png')
    cv2.imwrite(occ_im_path, occ_im)

    ellipse_im_path = os.path.join(output_dir, str(image_ind) + '_ellipse_im.png')
    cv2.imwrite(ellipse_im_path, ellipse_im)

def fit_ellipse(data_dir, RETR, APPROX, MAG, occ_thresh):
    clusters_path = os.path.join(data_dir, 'associations', 'clusters.json')
    if not os.path.exists(clusters_path):
        raise RuntimeError('clusters path does not exist: ' + clusters_path)
    
    ellipse_dir = os.path.join(data_dir, 'ellipses')
    if not os.path.exists(ellipse_dir):
        os.mkdir(ellipse_dir)

    clusters = read_json(clusters_path)

    path_ret = get_paths(data_dir, indices=None, use_filter_segs=True, include_cloud=True, use_filter_indices=True)
    image_inds, left_paths, disp_paths, cam_info_paths, _, seg_paths, cloud_cam_paths, _ = path_ret

    ellipse_dict = {}
    for i in range(len(image_inds)):
        fit_ellipse_image(clusters, ellipse_dict, ellipse_dir,
                          image_inds[i], left_paths[i],
                          disp_paths[i], seg_paths[i], 
                          cam_info_paths[i],
                          cloud_cam_paths[i],
                          RETR, APPROX, MAG, occ_thresh)
        
    output_path = os.path.join(ellipse_dir, 'ellipses.pkl')
    write_pickle(output_path, ellipse_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--retr', type=str, default='EXTERNAL')
    parser.add_argument('--chain_approx', type=str, default='NONE')
    parser.add_argument('--mag', type=int, default=3)
    parser.add_argument('--occ_thresh', type=float, default=0.80)

    args = parser.parse_args()
    return args

#python3 9_fit_ellipse.py --data_dir $DATA_DIR

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    retr = get_retr(args.retr)
    chain_approx = get_chain_approx(args.chain_approx)
    mag = args.mag
    occ_thresh = args.occ_thresh

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    fit_ellipse(data_dir, retr, chain_approx, mag, occ_thresh)
