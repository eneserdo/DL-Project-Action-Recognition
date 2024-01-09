#!/bin/bash

rosbag record -o $1 /zed/zed_node/confidence/confidence_map /zed/zed_node/depth/camera_info /zed/zed_node/depth/depth_registered /zed/zed_node/rgb/camera_info /zed/zed_node/rgb/image_rect_color /tf