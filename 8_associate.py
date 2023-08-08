import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/Highly-Connected-Subgraphs-Clustering-HCS')

import argparse
import os
import numpy as np
import scipy.optimize
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import distinctipy

from nbv_utils import get_paths, read_json, get_node_id
from nbv_utils import read_pickle, create_point_cloud, write_json, get_rot_trans_inv

from hcs import HCS

def process_image(left_path, disp_path, cam_info_path, 
                  trans_path, seg_path, cloud_cam_path,
                  image_ind, 
                  sphere_centre, sphere_radius,
                  args):
    
    cam_points = np.load(cloud_cam_path)
    transform_path = cloud_cam_path.replace('.npy', '.pkl')
    R_0, t_0 = read_pickle(transform_path)

    colors = cv2.imread(left_path)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    colors = (colors.astype(float)) / 255

    R_0_inv, t_0_inv = get_rot_trans_inv(R_0, t_0)

    segmentations = read_pickle(seg_path)
    
    fruitlets = []

    for i in range(len(segmentations)):
        seg_inds, _ = segmentations[i]

        seg_points = cam_points[seg_inds[:, 0], seg_inds[:, 1]]

        good_seg_points = ~np.isnan(seg_points).any(axis=1)
        seg_points = seg_points[good_seg_points]

        #convert sphere centre to cam
        sphere_centre_cam = (R_0_inv @ sphere_centre.T).T + t_0_inv

        good_seg_points = np.linalg.norm((seg_points - sphere_centre_cam), axis=1) < sphere_radius
        seg_points = seg_points[good_seg_points]

        if seg_points.shape[0] == 0:
            continue

        if args.use_mean:
            centroid = np.mean(seg_points, axis=0)
        else:
            centroid = np.median(seg_points, axis=0)

        fruitlets.append((image_ind, i, centroid))

    if len(fruitlets) == 0:
        raise RuntimeError('No fruitlets for: ' + left_path)

    return fruitlets, cam_points, colors, R_0, t_0

def vis_centroids(cloud_path, fruitlets, R_0, t_0, t_new=[0, 0, 0]):
    centroids = []
    colors = []
    color = [1.0, 0.0, 0.0]

    for _, _, centroid in fruitlets:
        centroids.append(centroid)
        colors.append(color)

    centroids = np.vstack(centroids)
    colors = np.vstack(colors)

    centroids = (R_0 @ (centroids + t_new).T).T + t_0
    
    create_point_cloud(cloud_path, centroids, colors)

def vis_res(cloud_path, points, colors, R_0, t_0, t):
    points = points.reshape((-1, 3))
    colors = colors.reshape((-1, 3))

    points = (R_0 @ (points + t).T).T + t_0

    create_point_cloud(cloud_path, points, colors)

def get_params():
    tx, ty, tz = 0, 0, 0
    
    return [tx, ty, tz]

def process_params(params, param_ind, base_image_ind):
    params_n = params[3*param_ind:3*(param_ind + 1)]
    tx, ty, tz = params_n
    t = np.array([tx, ty, tz])

    if param_ind == base_image_ind:
        return [0, 0, 0]

    return t

def get_centroids(im_fruitlets, include_details=False):
    centroids = []
    image_inds = []
    seg_inds = []
    for image_ind, seg_ind, centroid in im_fruitlets:
        centroids.append(centroid)

        image_inds.append(image_ind)
        seg_inds.append(seg_ind)

    centroids = np.vstack(centroids)

    if include_details:
        return centroids, image_inds, seg_inds
    else:
        return centroids

def compute_residuals(X, Y, VI=np.eye(3)):
    dist = cdist(X, Y, 'mahalanobis', VI=VI).flatten()
    return dist

def residual(params, fruitlets, full_Rs, full_ts, base_image_ind):
    num_fruitlets = len(fruitlets)
    errors = []
    for ind_0 in range(num_fruitlets):
        t_0 = process_params(params, ind_0, base_image_ind)
        c_0 = get_centroids(fruitlets[ind_0])
        
        R_orig_0 = full_Rs[ind_0]
        t_orig_0 = full_ts[ind_0]

        c_0 = (R_orig_0 @ (c_0 + t_0).T).T + t_orig_0

        for ind_1 in range(ind_0 + 1, num_fruitlets):
            t_1 = process_params(params, ind_1, base_image_ind)
            c_1 = get_centroids(fruitlets[ind_1])

            R_orig_1 = full_Rs[ind_1]
            t_orig_1 = full_ts[ind_1]

            c_1 = (R_orig_1 @ (c_1 + t_1).T).T + t_orig_1

            error = compute_residuals(c_0, c_1)
            errors = np.concatenate((errors, error))

    return errors

def fruitlet_associate(fruitlets, full_Rs, full_ts, base_image_ind,
                       max_trans, f_scale):
    num_images = len(fruitlets)

    params = []
    lower_bounds = []
    upper_bounds = []
    for _ in range(num_images):
        params = params + get_params()
        lower_bounds = lower_bounds + [-max_trans[0], -max_trans[1], -max_trans[2]]
        upper_bounds = upper_bounds + [max_trans[0], max_trans[1], max_trans[2]]
        
    res = scipy.optimize.least_squares(residual, params, 
                                       bounds=(lower_bounds, upper_bounds),
                                       #method='lm',
                                       #loss='huber',
                                       loss='arctan',
                                       f_scale=f_scale,
                                       #max_nfev = 50000,
                                       args=(fruitlets, full_Rs, full_ts, base_image_ind,))
    
    ret_vals = []
    for param_ind in range(num_images):
        t = process_params(res.x, param_ind, base_image_ind)

        ret_vals.append(t)

    return res.success, ret_vals

def add_edges(nodes_0, nodes_1, G, max_dist, node_set):
    c_0, image_inds_0, seg_inds_0 = nodes_0
    c_1, image_inds_1, seg_inds_1 = nodes_1

    num_0 = c_0.shape[0]
    num_1 = c_1.shape[0]

    for ind_0 in range(num_0):
        node_id_0 = get_node_id(image_inds_0[ind_0], seg_inds_0[ind_0])
        G.add_node(node_id_0)
        node_set.add(node_id_0)

        for ind_1 in range(num_1):
            node_id_1 = get_node_id(image_inds_1[ind_1], seg_inds_1[ind_1])
            G.add_node(node_id_1)
            node_set.add(node_id_1)

            dist = np.linalg.norm(c_0[ind_0] - c_1[ind_1])
            if dist > max_dist:
                continue

            G.add_edge(node_id_0, node_id_1)

def cluster(fruitlets, full_Rs, full_ts, opt_ret_vals, max_dist):
    num_fruitlets = len(fruitlets)

    #build the graph
    G = nx.Graph()
    node_set = set()
    for ind_0 in range(num_fruitlets):
        t_0 = opt_ret_vals[ind_0]
        c_0, image_inds_0, seg_inds_0 = get_centroids(fruitlets[ind_0], include_details=True)
        
        R_orig_0 = full_Rs[ind_0]
        t_orig_0 = full_ts[ind_0]

        c_0 = (R_orig_0 @ (c_0 + t_0).T).T + t_orig_0

        for ind_1 in range(ind_0 + 1, num_fruitlets):
            t_1 = opt_ret_vals[ind_1]
            c_1, image_inds_1, seg_inds_1 = get_centroids(fruitlets[ind_1], include_details=True)

            R_orig_1 = full_Rs[ind_1]
            t_orig_1 = full_ts[ind_1]

            c_1 = (R_orig_1 @ (c_1 + t_1).T).T + t_orig_1

            add_edges((c_0, image_inds_0, seg_inds_0), 
                        (c_1, image_inds_1, seg_inds_1), 
                        G, max_dist, node_set)

    #highly connected subgraph
    #TODO clusters with 2 or fewer detections will be removed
    #maybe somehow make this so that it's 1 or fewer?
    #might not need, could maybe change the quality from > to >=
    #TODO what do we do if spurious detection in an image? how does that filter out again?
    #ie two nodes from same image in final cluster
    G_HCS = nx.Graph()
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for sub_graph in sub_graphs:
        sub_graph = HCS(sub_graph)

        G_HCS.add_edges_from(sub_graph.edges())


    cluster_dict = {'clusters': {},
                    'fruitlets': {}}
    hcs_sub_graphs = (G.subgraph(c).copy() for c in nx.connected_components(G_HCS))
    for _class, _cluster in enumerate(hcs_sub_graphs):
        c = list(_cluster.nodes)

        cluster_dict['clusters'][_class] = c

        for node_id in c:
            if node_id in cluster_dict['fruitlets']:
                raise RuntimeError('node id appeared twice: ' + node_id)
            cluster_dict['fruitlets'][node_id] = _class

    return cluster_dict, G, G_HCS

def vis_cluster_segs(output_dir, image_inds, left_paths, seg_paths, cluster_dict):
    num_clusters = len(cluster_dict['clusters'])
    colors = distinctipy.get_colors(num_clusters)

    for i in range(len(image_inds)):
        image_ind = image_inds[i]
        left_path = left_paths[i]
        seg_path = seg_paths[i]

        im = cv2.imread(left_path)
        segmenations = read_pickle(seg_path)

        for seg_id in range(len(segmenations)):
            node_id = get_node_id(image_ind, seg_id)
            if not node_id in cluster_dict['fruitlets']:
                continue

            cluster_id = cluster_dict['fruitlets'][node_id]
            color = colors[cluster_id]
            color = ([int(255*color[0]), int(255*color[1]), int(255*color[2])])

            #just for vis so not worrying about filtering seg_inds
            #for invalid point cloud points
            seg_inds, _ = segmenations[seg_id]

            im[seg_inds[:, 0], seg_inds[:, 1]] = color

        output_path = os.path.join(output_dir, str(image_ind) + '.png')
        cv2.imwrite(output_path, im)


def associate(data_dir, args):
    associations_dir = os.path.join(data_dir, 'associations')
    if not os.path.exists(associations_dir):
        os.mkdir(associations_dir)

    path_ret = get_paths(data_dir, indices=None, use_filter_segs=True, 
                         use_filter_indices=True, include_cloud=True)
    image_inds, left_paths, disp_paths, cam_info_paths, trans_paths, seg_paths, cloud_cam_paths, _ = path_ret

    base_fruitlet_path = os.path.join(data_dir, 'base_fruitlet.json')
    base_fruilet = read_json(base_fruitlet_path)
    sphere_centre = np.array(base_fruilet['centre'])
    sphere_radius = base_fruilet['radius']

    full_fruitlets = []
    full_Rs = []
    full_ts = []
    full_cam_points = []

    base_image_ind = None
    max_fruitlets = -1
    for i in range(len(left_paths)):
        im_fruitlets, im_cam_points, im_colors, R_0, t_0 = process_image(left_paths[i], disp_paths[i], 
                                                      cam_info_paths[i], trans_paths[i], 
                                                      seg_paths[i], cloud_cam_paths[i],
                                                      image_inds[i], 
                                                      sphere_centre, sphere_radius,
                                                      args)
        
        vis_path = os.path.join(associations_dir, str(image_inds[i]) + '_centroid.pcd')
        vis_centroids(vis_path, im_fruitlets, R_0, t_0)

        full_fruitlets.append(im_fruitlets)
        full_Rs.append(R_0)
        full_ts.append(t_0)
        full_cam_points.append((im_cam_points, im_colors))

        if (len(im_fruitlets) > max_fruitlets):
            max_fruitlets = len(im_fruitlets)
            base_image_ind = i

    if base_image_ind is None:
        raise RuntimeError('No base image ind')
    
    success, ret_vals = fruitlet_associate(full_fruitlets, full_Rs, full_ts, base_image_ind,
                                           [args.max_xy_trans, args.max_xy_trans, args.max_z_trans], 
                                           args.f_scale)

    if success:
        print('opt success')
    else:
        print('opt failed')

    for i in range(len(left_paths)):
        t = ret_vals[i]
        R_0 = full_Rs[i]
        t_0 = full_ts[i]
        cam_points, colors = full_cam_points[i]

        vis_path = os.path.join(associations_dir, str(image_inds[i]) + '_trans_wold_cloud.pcd')
        vis_res(vis_path, cam_points, colors, R_0, t_0, t)

        vis_path = os.path.join(associations_dir, str(image_inds[i]) + '_trans_centroid.pcd')
        vis_centroids(vis_path, full_fruitlets[i], R_0, t_0, t_new=t)

        transform_path = os.path.join(associations_dir, str(image_inds[i]) + '_transform.npy')
        np.save(transform_path, t)

    cluster_dict, G, G_HCS = cluster(full_fruitlets, full_Rs, full_ts, ret_vals, args.cluster_max_dist)

    g_path = os.path.join(associations_dir, 'pre_cluster_graph.png')
    nx.draw(G)
    plt.savefig(g_path)
    
    plt.clf()
    hcs_g_path = os.path.join(associations_dir, 'post_cluster_graph.png')
    nx.draw(G_HCS)
    plt.savefig(hcs_g_path)
    
    cluster_path = os.path.join(associations_dir, 'clusters.json')
    write_json(cluster_path, cluster_dict, pretty=True)

    #just for some nice visualizations
    cluster_seg_vis_dir = os.path.join(associations_dir, 'fruitlet_vis')
    if not os.path.exists(cluster_seg_vis_dir):
        os.mkdir(cluster_seg_vis_dir)

    vis_cluster_segs(cluster_seg_vis_dir, image_inds, left_paths, seg_paths, cluster_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--use_mean', action='store_false')

    parser.add_argument('--max_z_trans', type=float, default=0.01)
    parser.add_argument('--max_xy_trans', type=float, default=0.01)

    parser.add_argument('--f_scale', type=float, default=0.007)
    parser.add_argument('--cluster_max_dist', type=float, default=0.007)

    args = parser.parse_args()
    return args

#python3 8_associate.py --data_dir $DATA_DIR

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)
    
    associate(data_dir, args)