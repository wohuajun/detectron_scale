#! /usr/bin/env python
# -*- coding:UTF-8-*-

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Construct minibatches for Detectron networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np
import os
import math
cur_path = os.path.dirname(os.path.abspath(__file__))
# print(cur_path)
import sys
sys.path.append(cur_path)
from detectron.core.config import cfg
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.retinanet as retinanet_roi_data
import detectron.roi_data.rpn as rpn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
from augment import *
#from augment import crop
#from augment import creat_new_roidb
import random
import copy
import time
from rotation import *

logger = logging.getLogger(__name__)

# crop裁剪图片是的长宽占比
WIDTH_REGION = cfg.TRAIN.CROP_WIDTH 
HEIGHT_REGION = cfg.TRAIN.CROP_HEIGHT 
# 当对MaskRcnn开启时需要对实例分割点进行变换
MASKRCNN_SWITCH = cfg.TRAIN.MASKRCNN_SWITCH


def get_minibatch_blob_names(is_training=True):
    #按照数据加载器(data loader)读取的顺序返回数据blob的name
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN blob_names
        blob_names += rpn_roi_data.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        blob_names += retinanet_roi_data.get_retinanet_blob_names(
            is_training=is_training
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += fast_rcnn_roi_data.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb):
    """载入roidb,构建一个minibatch采样."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    # 生成空字典数据  如  {'data': [], 'fpn_labels_int32_wide_fpn1': []}等
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = rpn_roi_data.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        im_width, im_height = im_blob.shape[3], im_blob.shape[2]
        # im_width, im_height corresponds to the network input: padded image
        # (if needed) width and height. We pass it as input and slice the data
        # accordingly so that we don't need to use SampleAsOp
        valid = retinanet_roi_data.add_retinanet_blobs(
            blobs, im_scales, roidb, im_width, im_height
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid

# 生成caffe2要用到的数据
def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    #print("qwryyudfggjhgkjgk{}sdfgfdsg".format(roidb))
    # print("minibatch.py 118 此轮roidb的长度为:",len(roidb))
    # 获得数据的数量
    num_images = len(roidb)
    # 在这批数据中最XXXX的采样随机比例   (600,)这个的len()只是1 np.random.randint即在第一个参数和'high'值-1,的整数间生成list,list元素数量是roidb的数量
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )                                                         # 因为cfg.TRAIN.SCALES为(600,), 所以生成的是一个全为0的list
    processed_ims = []                                        # 图片数据的list
    im_scales = []                                            # 尺度比例list
    for i in range(num_images):

        try:
            im = cv2.imread(roidb[i]['image'])                             # 读取roidb的图像数据

        except Exception as e:
            print("Exception:",e)
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])    # 如果图像不存在则assert
        if roidb[i]['flipped']:                                        # 如果图的flipped为翻转,则翻转 
            im = im[:, ::-1, :]                                        # 对图像的宽做倒序  即x轴flipped
        # s_t1 = time.time()
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]                  # target_size是600        
        im, im_scale = blob_utils.prep_im_for_blob(                    # im是 resize后的数据, im_scale 是resize的比例  参数中cfg.PIXEL_MEANS/cfg.TRAIN.MAX_SIZE在配置文件中有配置
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale)                                     # 加入到im_scales的list中
        processed_ims.append(im)                                       # 加入到图像数据list中

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)
    #e_t1 = time.time()
    #t_t1 = e_t1 - s_t1
    #print("时间_最终：{}".format(t_t1))
    return blob, im_scales

"""Construct minibatches for Detectron networks."""

def get_minibatch_mul(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    #print("minibatch.py 157 ---------{}".format(len(roidb)))
    # Get the input image blob, formatted for caffe2
    im_blob, im_scales, roidb_ = _get_img_blob_mul(roidb)
    blobs['data'] = im_blob
    if cfg.RPN.RPN_ON:
    #     # RPN-only or end-to-end Faster/Mask R-CNN
        valid = rpn_roi_data.add_rpn_blobs(blobs, im_scales, roidb_)
        return blobs,valid
    if cfg.RETINANET.RETINANET_ON:
        im_width, im_height = im_blob.shape[3], im_blob.shape[2]
        # im_width, im_height corresponds to the network input: padded image
        # (if needed) width and height. We pass it as input and slice the data
        # accordingly so that we don't need to use SampleAsOp
        valid = retinanet_roi_data.add_retinanet_blobs(
            blobs, im_scales, roidb_, im_width, im_height
        )
    else:
        # Fast R-CNN like models trained on
        # precomputed proposals
        valid = fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb_)
    return blobs, valid


def _get_img_blob_mul(roidb):

    num_images = len(roidb)
    processed_img=[]
    imgs_scale=[]
    scale_inds=np.random.randint(0,len(cfg.TRAIN.SCALES),size=num_images)
    all_roidb=[]

    for i in range(num_images):
        img_path=roidb[i]['image']
        img=cv2.imread(img_path)
        h,w = img.shape[:2]
        # 如果图的flipped为翻转,则翻转 
        if roidb[i]['flipped']:                                       
            img = img[:, ::-1, :] 
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        # im是 resize后的数据, im_scale 是resize的比例  参数中cfg.PIXEL_MEANS/cfg.TRAIN.MAX_SIZE在配置文件中有配置
        im, imscale = blob_utils.prep_im_for_blob(                    
            img, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        processed_img.append(im)
        imgs_scale.append(imscale)                             
        all_roidb.append(roidb[i])

        img_ = img.copy()   # 其实在原图做完计算之后可以不用考虑值会改变的事了


        # print("copy图像时间：{}".format(t_t_))
        coor, new_boxes_crop, img_crop_w, img_crop_h, index_list =  crop(h, w, roidb[i], WIDTH_REGION, HEIGHT_REGION)

        #  深拷贝以避免改变原始图像数据
        img_ = copy.deepcopy(img_[coor[0]:coor[1],coor[2]:coor[3]])



        if not new_boxes_crop  == []:
            new_roidb = creat_new_roidb(roidb[i], img_crop_w, img_crop_h, new_boxes_crop, index_list)

            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, imscale = blob_utils.prep_im_for_blob(img_, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            processed_img.append(im)
            imgs_scale.append(imscale)
            all_roidb.append(new_roidb)

        else:

            roidb_copy = creat_new_roidb_empty(roidb[i], img_crop_w, img_crop_h)

            target_size=cfg.TRAIN.SCALES[scale_inds[i]]
            im, imscale = blob_utils.prep_im_for_blob(img_, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)

            processed_img.append(im)
            imgs_scale.append(imscale)
            all_roidb.append(roidb_copy)


    blob=blob_utils.im_list_to_blob(processed_img) 

    return blob, imgs_scale, all_roidb

