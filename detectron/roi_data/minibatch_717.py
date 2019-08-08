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
cur_path = os.path.dirname(os.path.abspath(__file__))
print(cur_path)
import sys
sys.path.append(cur_path)
from detectron.core.config import cfg
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.retinanet as retinanet_roi_data
import detectron.roi_data.rpn as rpn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
from augment import crop
import random

logger = logging.getLogger(__name__)


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
    #print("sdfgfdgfghrdfh!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", roidb[0])
    # with open("roidb.txt", "w") as  f:
    #    f.write(roidb[1])
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
    print("~~~~~此轮roidb的长度为:",len(roidb))
    # 获得数据的数量
    num_images = len(roidb)
    # 在这批数据中最XXXX的采样随机比例   (600,)这个的len()只是1 np.random.randint即在第一个参数和'high'值-1,的整数间生成list,list元素数量是roidb的数量
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )                                                         # 因为cfg.TRAIN.SCALES为(600,), 所以生成的是一个全为0的list
    processed_ims = []                                        # 图片数据的list
    im_scales = []                                            # 尺度比例list
    for i in range(num_images):
        print("minibatch.py 109 @@@@@@@roidb{}".format(roidb[i]['boxes']))
        try:
            im = cv2.imread(roidb[i]['image'])                             # 读取roidb的图像数据
            print("---这是此次的for循环的第{}/{}轮,图像为{}".format(i,num_images,roidb[i]['image']))
        except Exception as e:
            print("Exception:",e)
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])    # 如果图像不存在则assert
        if roidb[i]['flipped']:                                        # 如果图的flipped为翻转,则翻转 
            im = im[:, ::-1, :]                                        # 对图像的宽做倒序  即x轴flipped
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]                  # target_size是600        
        im, im_scale = blob_utils.prep_im_for_blob(                    # im是 resize后的数据, im_scale 是resize的比例  参数中cfg.PIXEL_MEANS/cfg.TRAIN.MAX_SIZE在配置文件中有配置
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale)                                     # 加入到im_scales的list中
        processed_ims.append(im)                                       # 加入到图像数据list中

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

"""Construct minibatches for Detectron networks."""

def get_minibatch_mul(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    im_blob, im_scales, roidb_ = get_img_blob_mul(roidb)
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



def _mat_inter(box1,box2):

    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)

    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)


    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:

        return True
    else:
        return False

def _solve_coincide(box1,box2):
    # box=(xA,yA,xB,yB)

    if _mat_inter(box1,box2)==True:

        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        coincide=intersection/(area2+0.00001)#(area1+area2-intersection)
        return coincide

    else:
        return False

def compute_bbox_regression_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = np.array(entry['boxes'])
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((np.shape(rois)[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
    #     # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]
    #
    # # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))
    #
    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def get_img_blob_mul(roidb):
    num_images = len(roidb)
    

    processed_img=[]
    imgs_scale=[]
    roidb_base={}
    all_roidb=[]
    scale_inds=np.random.randint(0,len(cfg.TRAIN.SCALES),size=num_images)
    
    for i in range(num_images):
        # print(i)
        roidb_copy= {}
        # print("minibatch.py  220 @@@@@@@roidb[boxes]{}".format(roidb[i]['boxes']))
        img_path=roidb[i][u'image']
        # print("minibatch.py  222  image:------------------",img_path)
        img=cv2.imread(img_path)
        h,w = img.shape[:2]
        # 如果图的flipped为翻转,则翻转 
        if roidb[i]['flipped']:                                       
            img = img[:, ::-1, :] 
        # target_size是600,或者yaml中设置的尺度
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        # im是 resize后的数据, im_scale 是resize的比例  参数中cfg.PIXEL_MEANS/cfg.TRAIN.MAX_SIZE在配置文件中有配置
        im, imscale = blob_utils.prep_im_for_blob(                    
            img, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        processed_img.append(im)
        imgs_scale.append(imscale)                             
        all_roidb.append(roidb[i])

        z=crop(h,w,roidb[i],[0.8, 0.9],[0.8, 0.9])
        coor, new_boxes_crop, new_classes_crop, img_crop_w, img_crop_h, index_list=z

        if not new_boxes_crop  == []:
            # print("minibatch   237   ",index_list)
            gt_class=roidb[i]['gt_classes'][index_list]
            # print("minibatch   239   ",gt_class)
            gt_overlaps=roidb[i]['gt_overlaps'][index_list]
            max_overlaps=  roidb[i]['max_overlaps'][index_list]   #np.max(gt_overlaps,axis=1)
            max_classes= roidb[i]['max_classes'][index_list]              #np.argmax(gt_overlaps,axis=1)

            roidb_copy = roidb[i]
            roidb_copy['boxes']=np.array(new_boxes_crop, dtype=np.float32)
            roidb_copy[u'width']=img_crop_w
            roidb_copy[u'height']=img_crop_h
            roidb_copy['gt_classes']=gt_class
            roidb_copy['gt_overlaps']=gt_overlaps
            roidb_copy['is_crowd']=roidb[i]['is_crowd'][index_list]
            roidb_copy['box_to_gt_ind_map']=roidb[i]['box_to_gt_ind_map'][index_list]
            roidb_copy[u'max_classes']=max_classes
            # print(roidb_copy[-1]['max_overlaps'])
            roidb_copy['max_overlaps']=max_overlaps
            # m=roidb_copy[-1]['max_overlaps']
            box_target=compute_bbox_regression_targets(roidb_copy)

            roidb_copy['bbox_target']=box_target
            target_size=cfg.TRAIN.SCALES[scale_inds[i]]
            im,imscale=blob_utils.prep_im_for_blob(img[coor[0]:coor[1],coor[2]:coor[3]],cfg.PIXEL_MEANS,target_size,cfg.TRAIN.MAX_SIZE)
            processed_img.append(im)
            imgs_scale.append(imscale)
            all_roidb.append(roidb_copy)
        else:
            roidb_copy = roidb[i]
            roidb_copy[u'width']=img_crop_w
            roidb_copy[u'height']=img_crop_h
            target_size=cfg.TRAIN.SCALES[scale_inds[i]]
            im,imscale=blob_utils.prep_im_for_blob(img[coor[0]:coor[1],coor[2]:coor[3]],cfg.PIXEL_MEANS,
                                                   target_size,cfg.TRAIN.MAX_SIZE)
            processed_img.append(im)
            imgs_scale.append(imscale)
            all_roidb.append(roidb_copy)
    # if not processed_img==[]:
    blob=blob_utils.im_list_to_blob(processed_img) 
        # print(" minibatch  266 @@@@@@@imgs_scale{}, roidb[boxes]{}]".format(imgs_scale,all_roidb))
    return blob,imgs_scale, all_roidb