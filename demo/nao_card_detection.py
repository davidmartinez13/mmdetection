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

    # img = mmcv.imconvert(img, 'bgr', 'rgb')
    # visualizer.add_datasample(
    #     name='result',
    #     image=img,
    #     data_sample=result,
    #     draw_gt=False,
    #     pred_score_thr=args.score_thr,
    #     show=False)

    # img = visualizer.get_image()
    for bbox, score, label_torch in zip(bboxes, scores, labels):
        if score < args.score_thr:
            continue
        label = int(label_torch)
        print(label)
        color = get_label_color(label)

        bbox = list(map(int, bbox))
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        label_text = f'{get_label_name(label)} | Score: {score:.2f}'  # Use the label name
        cv2.putText(img, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # img = mmcv.imconvert(img, 'bgr', 'rgb')
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

