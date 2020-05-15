#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   : 测试预测
@File    :   pred_gen.py    
@Author  : vincent
@Time    : 2020/5/15 10:41 上午
@Version : 1.0 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import glob

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'table')


def get_files(data_path):
    """
    获取目录下以及子目录下的图片
    :param data_path:
    :return:
    """
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG','bmp']
    for ext in exts:
        # glob.glob 得到所有文件名
        # 一层 2层子目录都取出来
        files.extend(glob.glob(os.path.join(data_path, '*.{}'.format(ext))))
        files.extend(glob.glob(os.path.join(data_path, '*', '*.{}'.format(ext))))
    return files


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # TODO 不管什么图都是给出300个结果？ 然后根据概率筛选？去掉重叠？
    #
    print("图片预测结果：",boxes.shape,boxes[0],scores[0])
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8 #
    NMS_THRESH = 0.3 #
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        #TODO 4-8?  想只筛选出这个类别的box，但是逻辑没看懂
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        print("cls_boxes:",cls_boxes.shape,cls_boxes[0])
        cls_scores = scores[:, cls_ind]
        print("cls_scores:",cls_scores.shape,cls_scores[0])

        # 合成box和框
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #TODO NMS筛选之后剩下的
        keep = nms(dets, NMS_THRESH)
        print("keep:",keep)
        dets = dets[keep, :]
        print("dets:",dets)
        print("cls:",cls)

        # TODO write output
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    tfmodel = "output/res101/gen_train/default/res101_faster_rcnn_iter_70000.ckpt"

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)

    net = resnetv1(num_layers=101)

    # # load model
    # net.create_architecture("TEST", imdb.num_classes, tag='default',
    #                         anchor_scales=cfg.ANCHOR_SCALES,
    #                         anchor_ratios=cfg.ANCHOR_RATIOS)

    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    input_path = "data/pred/input1"
    #TODO
    im_names = get_files(input_path)
    print("加载的图片：",im_names)
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(im_name))
        demo(sess, net, im_name)
