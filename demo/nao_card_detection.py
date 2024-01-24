#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector
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

def image_callback(msg):
    img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    img_h, img_w = img.shape[0:2]

    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=args.score_thr,
        show=False)

    img = visualizer.get_image()
    img = mmcv.imconvert(img, 'bgr', 'rgb')
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

