# This code was used to extract RGB and Depth images from a rosbag file. 
# To use this code, it must be compiled with ROS. You can not use this code without ROS

# I mostly stole from: https://gist.github.com/rohanpsingh/9ac99c46aef8ccb618cdad18cd20e068#file-bag_to_png-py-L1-L69

import argparse
import datetime
import os
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
import rosbag
from sensor_msgs.msg import Image


def main(args):

    if not os.path.exists(args.input_bag_file):
        print(f"Couldn't find bag file. check path: {args.input_bag_file}")
        sys.exit(-1)

    if not os.path.exists(args.output_dir):
        print(f"Couldn't find output directory. check path: {args.output_dir}")
        sys.exit(-1)


    # create folder name based on bag file name and date
    bag_file_name = os.path.basename(args.input_bag_file)[:-4]
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = os.path.join(args.output_dir, bag_file_name + "_" + date)
    os.mkdir(out_dir)

    out_rgb_dir = os.path.join(out_dir, "rgb")
    out_depth_dir = os.path.join(out_dir, "depth")
    os.mkdir(out_rgb_dir)
    os.mkdir(out_depth_dir)

    
    print("Reading", args.input_bag_file)

    bag = rosbag.Bag(args.input_bag_file)
    available_topics = bag.get_type_and_topic_info()[1].keys()
    
    print("RGB message count:", bag.get_message_count(args.rgb_topic))
    print("Depth message count:", bag.get_message_count(args.depth_topic))

    assert (
        args.rgb_topic in available_topics
    ), f"{args.rgb_topic} not found in bag file: {args.input_bag_file}"
    assert (
        args.depth_topic in available_topics
    ), f"{args.depth_topic} not found in bag file: {args.input_bag_file}"

    # cv2.namedWindow(args.rgb_topic)
    # cv2.namedWindow(dep_topic)

    rgb_id = 0
    depth_id = 0

    # to write timestamps to text file
    rgb_file = open(os.path.join(out_dir, "rgb.txt"), "w")
    depth_file = open(os.path.join(out_dir, "depth.txt"), "w")

    # TODO: Check if there is any missing frame

    cv_bridge = CvBridge()

    try:  # to make sure we close the bag and text files
        for topic, msg, t in bag.read_messages(topics=[args.rgb_topic, args.depth_topic]):
            print(".", end="", flush=True)

            fn = format(((rospy.rostime.Time.to_nsec(t) / 1e9) - 1580300000), ".6f")
            try:
                if topic == args.rgb_topic:
                    file_name = f"rgb_{rgb_id:0>5}.png"
                    cvimg = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    out_fn = os.path.join(out_rgb_dir, file_name)
                    rgb_file.write(fn + ", " + file_name + "\n")
                    cv2.imwrite(out_fn, cvimg)
                    rgb_id += 1
                    

                elif topic == args.depth_topic:
                    file_name = f"depth_{depth_id:0>5}.npy"
                    cvimg = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    # print(cvimg.shape)
                    # print(cvimg.dtype)
                    # Filter out invalid depth values
                    # # remove nans
                    cvimg = np.where(np.isnan(cvimg), 0, cvimg)

                    # # remove posinfs
                    max_val = cvimg[cvimg != np.inf].max()
                    cvimg = np.where(cvimg == np.inf, max_val, cvimg)

                    # # remove neginfs
                    min_val = cvimg[cvimg != -np.inf].min()
                    cvimg = np.where(cvimg == -np.inf, min_val, cvimg)

                    # exclude alpha channel
                    # cvimg = cvimg[:, :, :-1]

                    # save as numpy array
                    np.save(os.path.join(out_depth_dir, file_name), cvimg)

                    depth_file.write(fn + ", " + file_name + "\n")
                    depth_id += 1

                # cv2.imshow(topic, cvimg)
                # cv2.waitKey(10)
            except CvBridgeError as e:
                print(e)
                sys.exit(-1)

    finally:
        rgb_file.close()
        depth_file.close()
        bag.close()

    print("\nDone!\n")


if __name__ == "__main__":
    
    # for action in ["stir", "pour", "slide", "cut"]:
    #     for i in range(1, 6):
    #         main(f"/home/enerd/dl_dataset/{action}{i}.bag")
    # main("/home/enerd/dl_dataset/stir11.bag")
    
    argparse = argparse.ArgumentParser()
    argparse.add_argument("-i", "--input_bag_file", type=str, required=True)
    argparse.add_argument("-rgb", "--rgb_topic", type=str, default="/zed/zed_node/rgb/image_rect_color")
    argparse.add_argument("-dep", "--depth_topic", type=str, default="/zed/zed_node/depth/depth_registered")
    argparse.add_argument("-o", "--output_dir", type=str, default="/home/enerd/rosdata")

    main(argparse.parse_args())