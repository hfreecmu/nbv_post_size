import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet_2023/scripts/nbv')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from nbv_utils import read_json, write_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', required=True)
    parser.add_argument('--res_type', required=True)
    parser.add_argument('--is_sim', action='store_true')

    parser.add_argument('--use_z_score', action='store_false')
    parser.add_argument('--z_score', type=int, default=3)

    args = parser.parse_args()
    return args


title_dict = {
    'linear_sampler': "Linear Sample Size vs. REPL_STRING Size",
    'nag_fvp_viewpoint_planner': "NAG FVP Size vs. REPL_STRING Size",
    'random_sampler': "Random Sample Size vs. REPL_STRING Size",
    'roi_viewpoint_planner': "FVP Size vs. REPL_STRING Size",
    'rvp_low_res_viewpoint_planner': "RVP Size vs. REPL_STRING Size",
    'sm_fvp_low_res_viewpoint_planner': "SM FVP Size vs. REPL_STRING Size"
}

args = parse_args()

res_dir = args.res_dir
res_type = args.res_type
is_sim = args.is_sim
use_Z_score = args.use_z_score
Z_score = args.z_score

if is_sim:
    REPL_TILE = "Ground Truth"
else:
    REPL_TILE = "Hand-Caliper"

title = title_dict[res_type].replace('REPL_STRING', REPL_TILE)

sizing_results_path = os.path.join(res_dir, 'sizing_results.json')
sizing_results = read_json(sizing_results_path)

csv_erros = []

unsized_cv_pcts = []
total_unsized = 0
total_num_fruitlets = 0

gt_sizes = []
cv_sizes = []
for tag_id in sizing_results['size_data']:
    num_cv_unsized = 0
    num_fruitlets = 0
    for fruitlet_id in sizing_results['size_data'][tag_id]:
        num_fruitlets += 1

        gt_size = sizing_results["size_data"][tag_id][fruitlet_id]['gt_size']
        cv_size = sizing_results["size_data"][tag_id][fruitlet_id]['cv_size']

        if gt_size == -1 and cv_size == -1:
            raise RuntimeError('something went wrong, double negative 1')
        
        #gt error
        if gt_size == -1:
            csv_erros.append('_'.join([tag_id, fruitlet_id]))
        elif cv_size == -1:
            num_cv_unsized += 1
        else:
            gt_sizes.append(gt_size)
            cv_sizes.append(cv_size)

        #TODO debug this
        #if cv_size > 20:
            #breakpoint()

    if num_fruitlets == 0:
        raise RuntimeError('0 fruitlets for ' + str(tag_id))
    
    total_unsized += num_cv_unsized
    total_num_fruitlets += num_fruitlets
    unsized_cv_pcts.append(num_cv_unsized / num_fruitlets)

mean_unsized_cv_pct = round(float(np.mean(unsized_cv_pcts)), 4)
total_unsized_cv_pct = round(total_unsized / total_num_fruitlets, 4)

unsized_cv_path = os.path.join(res_dir, 'unsized_cv_pct.json')
write_json(unsized_cv_path, {'mean_unsized_cv_pct': mean_unsized_cv_pct,
                             'total_unsized_cv_pct': total_unsized_cv_pct}, pretty=True)

gt_sizes = np.array(gt_sizes)
cv_sizes = np.array(cv_sizes)

#TODO no z_scores needed right now but maybe later
cv_mean = np.mean(cv_sizes)
cv_std = np.std(cv_sizes)
Z = (cv_sizes - cv_mean) / cv_std
if use_Z_score:
    good_inds = np.where(np.abs(Z) < Z_score)
    orig_sizes = gt_sizes.shape[0]
    gt_sizes = gt_sizes[good_inds]
    cv_sizes = cv_sizes[good_inds]
    new_sizes = gt_sizes.shape[0]
    print('num removed: ', orig_sizes - new_sizes)

mae = float(np.mean(np.abs(gt_sizes - cv_sizes)))
mape = float(np.mean(np.abs(gt_sizes - cv_sizes)/gt_sizes*100))
average_error_path = os.path.join(res_dir, 'average_errors.json')
write_json(average_error_path, {'mae': mae,
                                'mape': mape},
                                pretty=True)

for use_a in [True, False]:
    if use_a:
        b, a = np.polyfit(gt_sizes, cv_sizes, deg=1)
    else: 
        def f(x, b):
            return b*x
        a = 0
        popt, _ = curve_fit(f, gt_sizes, cv_sizes)
        b = popt[0]
    
    yfit = [a + b * xi for xi in gt_sizes]

    if a < 0:
        a_string = "{:.2f}".format(-a)
    else:
        a_string = "{:.2f}".format(a)
    b_string = "{:.2f}".format(b)
    label = b_string + '*x - ' + a_string

    r2_base = r2_score(cv_sizes, yfit)

    plt.scatter(gt_sizes, cv_sizes)
    plt.plot(gt_sizes, yfit, 'r', label=label)
    #plt.xlabel("Ground Truth Sizes (mm)")
    #plt.ylabel("Predicted Sizes (mm)")
    # plt.text(0.8, 1.01, 'r2 score: ' + "{:.3f}".format(r2_base), 
    #     fontsize=10, color='k',
    #     ha='left', va='bottom',
    #     transform=plt.gca().transAxes)

    #plt.title(title)
    #plt.legend(loc="upper left")
    
    if use_a:
        gt_cv_comp_path = os.path.join(res_dir, 'r2_offset.png')
        metrics_path = os.path.join(res_dir, 'r2_offset.json')
    else:
        gt_cv_comp_path = os.path.join(res_dir, 'r2_no_offset.png')
        metrics_path = os.path.join(res_dir, 'r2_no_offset.json')

    plt.savefig(gt_cv_comp_path)

    plt.close()

    metrics = {
        "r2": r2_base,
        "b": b_string,
        "a": a_string
    }

    write_json(metrics_path, metrics)