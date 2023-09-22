import argparse
import os
import rosbag

from nbv_utils import write_json, create_point_cloud, create_sphere_point_cloud

#assumes extract_nbv_images has been run with same args
valid_bag_types = ["tsdfroi", "dissim", "linear", "cluster"]
topic_name = "/target_centre"

def get_target_centre(bag_file):
    bag = rosbag.Bag(bag_file, "r")

    t_max = -1
    target_centre = None
    extracted_radius = None
    for _, msg, t in bag.read_messages(topics=[topic_name]):
        if t.to_sec() < t_max:
            continue

        t_max = t.to_sec()
        target_centre = msg.pose.position
        extracted_radius = msg.scale.x / 2

    if target_centre is not None:
        target_centre = [target_centre.x, target_centre.y, target_centre.z]

    return target_centre, extracted_radius

def extract_target_centres(input_dir, output_dir, bag_type, default_radius, use_extracted_radius, min_radius):
    for basename in os.listdir(input_dir):
        if not bag_type in basename:
            continue
        
        sub_input_dir = os.path.join(input_dir, basename)
        if not os.path.isdir(sub_input_dir):
            continue

        sub_output_dir = os.path.join(output_dir, basename)
        if not os.path.exists(sub_output_dir):
            print('No output dir for: ' + basename)
            continue

        bag_file = os.path.join(sub_input_dir, basename + '.bag')
        if not os.path.exists(bag_file):
            print('No bag file for: ' + basename)
            continue

        target_centre, extracted_radius = get_target_centre(bag_file)
        if target_centre is None:
            print('No target centre for: ' + basename)
            continue

        if use_extracted_radius:
            radius = extracted_radius
        else:
            radius = default_radius

        if radius < min_radius:
            radius = min_radius

        fruitlet_dict = {
            "radius": radius,
            'centre': target_centre
        }

        output_path = os.path.join(sub_output_dir, 'base_fruitlet.json')
        write_json(output_path, fruitlet_dict)

        sphere_points, sphere_colors = create_sphere_point_cloud(target_centre, radius, 1000)
        cloud_path = os.path.join(sub_output_dir, 'target_centre.pcd')
        create_point_cloud(cloud_path, sphere_points, sphere_colors)

        print('Done processing ' + basename)

#INPUT_DIR=PATH_TO_INPUT_DIR
#OUTPUT_DIR=PATH_TO_OUTPUT_DIR
#BAG_TYPE=TYPE_OF_BAG
#python3 extract_target_centres.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --bag_type $BAG_TYPE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bag_type', required=True)

    parser.add_argument('--default_radius', type=float, default=0.06)
    parser.add_argument('--min_radius', type=float, default=0.05)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    bag_type = args.bag_type
    default_radius = args.default_radius
    min_radius = args.min_radius

    if not os.path.exists(input_dir):
        raise RuntimeError('input_dir does not exist: ' + input_dir)
    
    if not os.path.exists(output_dir):
        raise RuntimeError('output_dir does not exist: ' + output_dir)
    
    if not bag_type in valid_bag_types:
        raise RuntimeError('Invalid bag_type: ' + bag_type + 
                           '. Choose one of ' + str(valid_bag_types) + '.')
    
    if bag_type == 'cluster':
        use_extracted_radius = True
    else:
        use_extracted_radius = False

    extract_target_centres(input_dir, output_dir, bag_type, default_radius, use_extracted_radius, min_radius)
    