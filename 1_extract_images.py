#!/usr/bin/env python2

import argparse
import os
import cv2
from threading import Lock
import yaml
import copy

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Empty
import rospy
from cv_bridge import CvBridge
import tf
import numpy as np

### WARNING WARNING WARNING
#needs use_sim_time set on ros_param server
#rosbag needs to be set in rosbag play
#how to fix error with empty ros message?

#steps:
#roscore
#rosparam set use_sim_time true
#python2 1_extract_images.py --data_dir ${OUTPUT_DIR} --process_rect_images --process_joints
#python2 1_extract_images.py --data_dir ${OUTPUT_DIR} --use_depth
#rosbag play -r ${RATE} --clock ${BAG_PATH}

class BagExtract():
    def __init__(self, data_dir, process_rect_images, process_joints, left_only, use_depth):
        rospy.init_node('bag_extractor', anonymous=True)

        self.data_dir = data_dir
        self.process_rect_images = process_rect_images
        self.process_joints = process_joints
        self.left_only = left_only
        self.use_depth = use_depth

        if not self.use_depth:
            self.raw_images_dir = os.path.join(data_dir, 'raw_images')

        self.camera_info_dir = os.path.join(data_dir, 'camera_info')

        if self.process_rect_images:
            self.rect_images_dir = os.path.join(data_dir, 'rect_images')

        if self.use_depth:
            self.depth_dir = os.path.join(data_dir, 'depth_images')

        self.message_num = 0

        if self.process_joints:
            self.ee_states_dir = os.path.join(data_dir, 'ee_states')
            self.tf_listener = tf.TransformListener()
            self.ee_transform_strings = []

        self.timestamps = []
        self.start_captures = []
        
        self.bridge = CvBridge()
        self.lock = Lock()

        self.world_frame = "/world"
        self.ee_frame = "/camera"

        self.make_dirs()

    def make_dirs(self):
        if self.use_depth:
            subdirs = [(self.camera_info_dir, True)]
        else:
            subdirs = [(self.raw_images_dir, True), (self.camera_info_dir, True)]
            
        if self.process_rect_images:
            include_right = not self.use_depth
            subdirs.append((self.rect_images_dir, include_right))

        for subdir, include_right in subdirs:
            if not os.path.exists(subdir):
                os.mkdir(subdir)

            left_dir = os.path.join(subdir, 'left')
            if not os.path.exists(left_dir):
                os.mkdir(left_dir)

            if (include_right) and (not self.left_only):
                right_dir = os.path.join(subdir, 'right')
                if not os.path.exists(right_dir):
                    os.mkdir(right_dir)

        if self.use_depth:
            if not os.path.exists(self.depth_dir):
                os.mkdir(self.depth_dir)

        if self.process_joints:
            if not os.path.exists(self.ee_states_dir):
                os.mkdir(self.ee_states_dir)

    def parse_camera_info_msg(self, msg):
        camera_info_dict = {
                'camera_matrix': {
                    'cols': 3,
                    'data': msg.K,
                    'rows': 3
                },
                'distortion_coefficients': {
                    'cols': 5,
                    'data': msg.D,
                    'rows': 1
                },
                'distortion_model': msg.distortion_model,
                'image_height': msg.height,
                'image_width': msg.width,
                'projection_matrix': {
                    'cols': 4,
                    'data': msg.P,
                    'rows': 3
                },
                'rectification_matrix': {
                    'cols': 3,
                    'data': msg.R,
                    'rows': 3
                },
                'header': {
                    'seq': msg.header.seq,
                    'stamp': {
                        'sec': msg.header.stamp.secs,
                        'nsec': msg.header.stamp.nsecs
                    },
                    'frame_id': msg.header.frame_id
                }
            }

        return camera_info_dict

    def process_transform(self, trans, rot):
        transform_dict = {
                        'quaternion': {
                            'qw': float(rot[3]),
                            'qx': float(rot[0]),
                            'qy': float(rot[1]),
                            'qz': float(rot[2])
                    },
                        'translation': {
                            'x': float(trans[0]),
                            'y': float(trans[1]),
                            'z': float(trans[2])
                            },
                    }

        transform_string = ', '.join([str(trans[0]), str(trans[1]), str(trans[2]), str(rot[0]), str(rot[1]), str(rot[2]), str(rot[3])])
        return transform_dict, transform_string

    def write_yaml(self, yaml_path, yml):
        with open(yaml_path, 'w+') as f:
            yaml.safe_dump(yml, f, default_flow_style=False)

    def process_images(self, left_image_raw_ros, right_image_raw_ros, 
                       left_camera_info_ros, right_camera_info_ros,
                       left_image_rect_ros=None, right_image_rect_ros=None,
                       depth_image_ros=None):

        self.lock.acquire()

        print('processing message: ' + str(self.message_num))
        
        if not self.use_depth:
            left_image_raw = self.bridge.imgmsg_to_cv2(left_image_raw_ros, desired_encoding='bgr8')

            if not self.left_only:
                right_image_raw = self.bridge.imgmsg_to_cv2(right_image_raw_ros, desired_encoding='bgr8')

        left_camera_info = self.parse_camera_info_msg(left_camera_info_ros)

        if not self.left_only:
            right_camera_info = self.parse_camera_info_msg(right_camera_info_ros)

        if self.process_rect_images:
            left_image_rect = self.bridge.imgmsg_to_cv2(left_image_rect_ros, desired_encoding='bgr8')
            if (not self.left_only) and (not self.use_depth):
                right_image_rect = self.bridge.imgmsg_to_cv2(right_image_rect_ros, desired_encoding='bgr8')

        if self.use_depth:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_ros, desired_encoding='passthrough')

        msg_string =  str(self.message_num).zfill(6)

        if not self.use_depth:
            left_image_raw_path = os.path.join(self.raw_images_dir, 'left', msg_string + '.png')
            right_image_raw_path = os.path.join(self.raw_images_dir, 'right', msg_string + '.png')

        left_camera_info_path = os.path.join(self.camera_info_dir, 'left', msg_string + '.yml')
        right_camera_info_path = os.path.join(self.camera_info_dir, 'right', msg_string + '.yml')

        if self.process_rect_images:
            left_image_rect_path = os.path.join(self.rect_images_dir, 'left', msg_string + '.png')
            if (not self.left_only) and not (self.use_depth):
                right_image_rect_path = os.path.join(self.rect_images_dir, 'right', msg_string + '.png')

        if self.use_depth:
            depth_image_path = os.path.join(self.depth_dir, msg_string + '.npy')
            depth_array = np.array(depth_image, dtype=np.float32)
            np.save(depth_image_path, depth_array)

        if not self.use_depth:
            cv2.imwrite(left_image_raw_path, left_image_raw)

            if not self.left_only:
                cv2.imwrite(right_image_raw_path, right_image_raw)

        self.write_yaml(left_camera_info_path, left_camera_info)

        if not self.left_only:
            self.write_yaml(right_camera_info_path, right_camera_info)

        if self.process_rect_images:
            cv2.imwrite(left_image_rect_path, left_image_rect)
            if (not self.left_only) and not (self.use_depth):
                cv2.imwrite(right_image_rect_path, right_image_rect)

        if self.process_joints:
            self.tf_listener.waitForTransform(self.world_frame, self.ee_frame, left_camera_info_ros.header.stamp, rospy.Duration(0.5))
            trans, rot = self.tf_listener.lookupTransform(self.world_frame, self.ee_frame, left_camera_info_ros.header.stamp)
            transform_dict, transform_string = self.process_transform(trans, rot)
            transform_path = os.path.join(self.ee_states_dir, msg_string + '.yml')
            self.write_yaml(transform_path, transform_dict)
            self.ee_transform_strings.append(transform_string)

        self.timestamps.append(str(left_camera_info_ros.header.stamp))

        self.message_num += 1
        self.lock.release()

    def process_images_left_only(self, left_image_raw_ros, left_camera_info_ros):

        raise RuntimeError('should not run')

        self.process_images(left_image_raw_ros, None,
                            left_camera_info_ros, None)

    def process_images_no_rect(self, left_image_raw_ros, right_image_raw_ros, 
                               left_camera_info_ros, right_camera_info_ros):
        
        raise RuntimeError('should not run')

        self.process_images(left_image_raw_ros, right_image_raw_ros,
                            left_camera_info_ros, right_camera_info_ros)

    def process_images_with_rect_left_only(self, left_image_raw_ros, 
                                 left_camera_info_ros,
                                 left_image_rect_ros):
        raise RuntimeError('should not run')

        self.process_images(left_image_raw_ros, None,
                            left_camera_info_ros, None,
                            left_image_rect_ros=left_image_rect_ros)

    def process_images_with_rect(self, left_image_raw_ros, right_image_raw_ros, 
                                 left_camera_info_ros, right_camera_info_ros,
                                 left_image_rect_ros, right_image_rect_ros):

        self.process_images(left_image_raw_ros, right_image_raw_ros,
                            left_camera_info_ros, right_camera_info_ros,
                            left_image_rect_ros=left_image_rect_ros,
                            right_image_rect_ros=right_image_rect_ros)

    def process_depth(self, left_camera_info_ros, right_camera_info_ros,
                            left_image_rect_ros,
                            depth_image_ros):
        self.process_images(None, None, left_camera_info_ros, right_camera_info_ros,
                            left_image_rect_ros, None, depth_image_ros)
    
    def get_process_image_callback(self):
        if self.use_depth:
            return self.process_depth

        if self.left_only and self.process_rect_images:
            return self.process_images_with_rect_left_only
        elif self.left_only:
            return self.process_images_left_only
        elif not self.process_rect_images:
            return self.process_images_no_rect
        else:
            return self.process_images_with_rect

    def process_start_capture_callback(self, msg):
        self.start_captures.append(str(rospy.Time.now()))

    def shutdown_hook(self):
        if self.process_joints:
            output_path = os.path.join(self.data_dir, 'world2effector_transform.txt')
            txt_data = '\n'.join(self.ee_transform_strings)
            with open(output_path, 'w') as f:
                f.write(txt_data)

        output_path = os.path.join(self.data_dir, 'timestamps.txt')
        txt_data = '\n'.join(self.timestamps)
        with open(output_path, 'w') as f:
            f.write(txt_data)

        output_path = os.path.join(self.data_dir, 'start_captures.txt')
        txt_data = '\n'.join(self.start_captures)
        with open(output_path, 'w') as f:
            f.write(txt_data)

def start(data_dir, process_rect_images, process_joints, left_only, use_depth, 
          queue_size=10, slop=0.3):
    bag_extractor = BagExtract(data_dir, process_rect_images, process_joints, left_only, use_depth)

    if not use_depth:
        left_image_topic = "theia/left/image_raw"
        right_image_topic = "theia/right/image_raw"

    left_camera_info_topic = "theia/left/camera_info"
    if not left_only:
        right_camera_info_topic = "theia/right/camera_info"

    if process_rect_images:
        left_image_rect_topic = "theia/left/image_rect_color"
        if (not left_only) and (not use_depth):
            right_image_rect_topic = "theia/right/image_rect_color"

    if use_depth:
        depth_topic = "depth_image"

    if not use_depth:
        left_image_sub = message_filters.Subscriber(left_image_topic, Image)

        if not left_only:
            right_image_sub = message_filters.Subscriber(right_image_topic, Image)
    
    left_camera_info_sub = message_filters.Subscriber(left_camera_info_topic, CameraInfo)

    if not left_only:
        right_camera_info_sub = message_filters.Subscriber(right_camera_info_topic, CameraInfo)
    
    if process_rect_images:
        left_image_rect_sub = message_filters.Subscriber(left_image_rect_topic, Image)
        if (not left_only) and (not use_depth):
            right_image_rect_sub = message_filters.Subscriber(right_image_rect_topic, Image)

    if use_depth:
        depth_image_sub = message_filters.Subscriber(depth_topic, Image)

    if use_depth:
        async_topics = [left_camera_info_sub, right_camera_info_sub]
    elif not left_only:
        async_topics = [left_image_sub, right_image_sub,
                        left_camera_info_sub, right_camera_info_sub]
    else:
        async_topics = [left_image_sub, left_camera_info_sub, right_camera_info_sub]

    if process_rect_images:
        async_topics.append(left_image_rect_sub)
        if (not left_only) and (not use_depth):
            async_topics.append(right_image_rect_sub)

    if use_depth:
        async_topics.append(depth_image_sub)

    start_capture_topic = "start_capture"
    start_capture_sub = rospy.Subscriber(start_capture_topic, Empty, bag_extractor.process_start_capture_callback, queue_size=2)

    ts = message_filters.ApproximateTimeSynchronizer(async_topics, queue_size, slop)
    ts.registerCallback(bag_extractor.get_process_image_callback())

    rospy.on_shutdown(bag_extractor.shutdown_hook)

    print('services initialized')

    rospy.spin()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--process_rect_images', action='store_true')
    parser.add_argument('--process_joints', action='store_true')
    parser.add_argument('--left_only', action='store_true')
    parser.add_argument('--use_depth', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    process_rect_images = args.process_rect_images
    process_joints = args.process_joints
    left_only = args.left_only
    use_depth = args.use_depth

    if use_depth:
        left_only = False
        process_rect_images = True
        process_joints = True

    if not os.path.exists(data_dir):
        raise RuntimeError('data_dir does not exist: ' + data_dir)

    start(data_dir, process_rect_images, process_joints, left_only, use_depth)