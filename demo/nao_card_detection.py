#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector, DetInferencer
from mmdet.registry import VISUALIZERS
import torch
import argparse
import cv2
import re
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import argparse
import cv2
import time
import re
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np

def find_major_color(mask_red, mask_green, mask_yellow, mask_blue):
    total_pixels = mask_red.size

    count_red = cv2.countNonZero(mask_red)
    count_green = cv2.countNonZero(mask_green)
    count_yellow = cv2.countNonZero(mask_yellow)
    count_blue = cv2.countNonZero(mask_blue)

    # Find the color with the maximum count
    max_count = max(count_red, count_green, count_yellow, count_blue)

    # Check if there is a major color and its percentage is above a threshold
    if max_count / total_pixels > 0.3:
        if max_count == count_red:
            return "Red"
        elif max_count == count_green:
            return "Green"
        elif max_count == count_yellow:
            return "Yellow"
        elif max_count == count_blue:
            return "Blue"

    return None

def show_color_masks(cv_hsvimage):
    # Red Color
    lower_red1 = np.array([0, 100, 100], np.uint8)
    upper_red1 = np.array([10, 255, 255], np.uint8)
    lower_red2 = np.array([160, 100, 100], np.uint8)
    upper_red2 = np.array([180, 255, 255], np.uint8)

    mask_red1 = cv2.inRange(cv_hsvimage, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(cv_hsvimage, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green Color
    lower_green = np.array([40, 40, 40], np.uint8)
    upper_green = np.array([90, 255, 255], np.uint8)
    mask_green = cv2.inRange(cv_hsvimage, lower_green, upper_green)

    # Blue Color
    lower_blue = np.array([90, 50, 50], np.uint8)
    upper_blue = np.array([150, 255, 255], np.uint8)
    mask_blue = cv2.inRange(cv_hsvimage, lower_blue, upper_blue)

    # Yellow Color
    lower_yellow = np.array([20, 100, 100], np.uint8)
    upper_yellow = np.array([40, 255, 255], np.uint8)
    mask_yellow = cv2.inRange(cv_hsvimage, lower_yellow, upper_yellow)

    return mask_red, mask_green, mask_blue, mask_yellow

# ... (Other imports and functions)
def get_label_color(label):
    label_colors = {
        0: (0, 0, 255),    # Red for "one"
        1: (0, 255, 0),    # Green for "two"
        2: (255, 0, 0),    # Blue for "three"
        3: (255, 255, 0),  # Yellow for "four"
        4: (0, 255, 255),  # Cyan for "five"
        5: (255, 0, 255),  # Magenta for "six"
        6: (0, 128, 255),  # Orange for "seven"
        7: (128, 0, 255),  # Purple for "eight"
        8: (255, 128, 0),  # Brown for "nine"
        9: (128, 128, 128),  # Gray for "zero"
        10: (0, 255, 128),   # Light Green for "plus2"
        11: (128, 0, 128),   # Dark Purple for "block"
        12: (255, 128, 128),  # Light Red for "reverse"
        13: (128, 255, 128),  # Light Green for "change"
        14: (128, 128, 255)   # Light Blue for "plus4"
        # Add more colors for additional classes as needed
    }

    return label_colors.get(label, (255, 255, 255))  # Default color: White for unknown label

def get_label_name(label):
    label_names = {
        0: "one",
        1: "two",
        2: "three",
        3: "four",
        4: "five",
        5: "six",
        6: "seven",
        7: "eight",
        8: "nine",
        9: "zero",
        10: "plus2",
        11: "block",
        12: "reverse",
        13: "change",
        14: "plus4"
        # Add more label names for additional classes as needed
    }

    return label_names.get(label, f'unknown_{label}')

def image_callback(msg):
    img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    img_h, img_w = img.shape[0:2]

    result = inference_detector(model, img)
    bboxes = result.pred_instances.bboxes
    scores = result.pred_instances.scores
    labels = result.pred_instances.labels
    # print(result.pred_instances.keys())
    detected_color = None

    for bbox, score, label_torch in zip(bboxes, scores, labels):
        if score < args.score_thr:
            continue
        bbox = list(map(int, bbox))
        label = int(label_torch)
        color = get_label_color(label)
        if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] < img_w and bbox[3] < img_h:
            # Extract the region within the bounding box
            roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Convert the region to HSV for color detection
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define color masks for red, green, yellow, and blue
            mask_red, mask_green, mask_blue, mask_yellow = show_color_masks(roi_hsv)

            # Use the find_major_color function to detect the dominant color
            detected_color = find_major_color(mask_red, mask_green, mask_yellow, mask_blue)
            print(detected_color)
            cv2.imshow('Red Mask', mask_red)
            cv2.imshow('Green Mask', mask_green)
            cv2.imshow('Blue Mask', mask_blue)
            cv2.imshow('Yellow Mask', mask_yellow)

            if detected_color != None and label != 'plus4' and label != 'change':
                # Update the label with the detected color
                label_text = f'{get_label_name(label)} | Color: {detected_color} | Score: {score:.2f}'
            else:
                # If color detection fails, use only the label and score
                label_text = f'{get_label_name(label)} | Score: {score:.2f}'

            detected_color = None

            bbox = list(map(int, bbox))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow('result', img)

    # cv2.imshow('Detection', cv2.resize(frame_origin, (frame_origin.shape[1] * 2, frame_origin.shape[0] * 2)))
    key = cv2.waitKey(10)
    if key == 27 or key == ord('q') or key == ord('Q'):
        rospy.signal_shutdown('User requested shutdown')

    # Additional processing or publishing can be done here

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    rospy.init_node('nao_card_detector')
    rospy.Subscriber("/nao_robot/camera/bottom/camera/image_raw", Image, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

