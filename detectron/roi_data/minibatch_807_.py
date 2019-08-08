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
import sys
sys.path.append(cur_path)
from detectron.core.config import cfg
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.retinanet as retinanet_roi_data
import detectron.roi_data.rpn as rpn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import random
import copy
import time
from augment import *
from rotation import *

logger = logging.getLogger(__name__)
# 多尺度相关的参数设置
# 是否开启多尺度
MULTI_COLOR = cfg.TRAIN.MULTI_COLOR
# 是否开启高斯模糊
GAUSSIAN = cfg.TRAIN.GAUSSIAN
# 高斯核大小
GAUSSIAN_KERNEL = cfg.TRAIN.GAUSSIAN_KERNEL
# 高斯Sigmax值
GAUSSIAN_SIGMAX = cfg.TRAIN.GAUSSIAN_SIGMAX

# 是否做亮度增强
CAB = cfg.TRAIN.CAB + cfg.TRAIN.GAUSSIAN 
ALPHA = cfg.TRAIN.ALPHA
GAMMA = cfg.TRAIN.GAMMA

# 旋转相关参数设置
# 是否旋转
ROTATION = cfg.TRAIN.ROTATION
# 旋转数据的概率占比
ROT = cfg.TRAIN.ROT
# False则转90,180,270,True为随机角度
ROT_RANDOM = cfg.TRAIN.ROT_RANDOM
# 如果ROT_RANDOM为True,则此设置控制旋转的角度范围
ROT_RANDOM_ANGLE = cfg.TRAIN.ROT_RANDOM_ANGLE
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
    #print("现在执行的是无扩增版本")
    blobs = {k: [] for k in get_minibatch_blob_names()}
    print("blobs is ",blobs)
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
    processed_img = []                                        # 图片数据的list
    im_scales = []  
    all_roidb=[]                                          # 尺度比例list
    for i in range(num_images):
        #print("minibatch.py 109   @@@@@@@   roidb的boxes{}".format(roidb[i]['boxes']))
        try:
            im = cv2.imread(roidb[i]['image'])                             # 读取roidb的图像数据
            img = im.copy()
            #print("---这是此次的for循环的第{}/{}轮,图像为{}".format(i,num_images,roidb[i]['image']))
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
        processed_img.append(im)                                       # 加入到图像数据list中

        all_roidb.append(roidb[i])

        #--------------------- 无多尺度下开启旋转模块 --------------------------------------
        img_ = img.copy()   # 其实在原图做完计算之后可以不用考虑值会改变的事了
        roidb_n = roidb[i]
        if ROTATION:  # 是否旋转
            rot_random =  random.uniform(0,1)
            if rot_random <= ROT:
                if ROT_RANDOM :
                    angle = random.randint(-ROT_RANDOM_ANGLE,ROT_RANDOM_ANGLE)
                else: 
                    angle = random.randint(1,4)
                    angle = angle * 90.0

                img_ , h_, w_, l, n_w, n_h = rotate_image(img_, angle, True)     # 裁剪完后的图
                angle = (angle / 180.0 * math.pi)

                roidb_n = creat_rotation_roidb(roidb_n, angle, l, w_, h_, n_w, n_h,  MASKRCNN_SWITCH)

            else: pass
        
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            #im, imscale = blob_utils.prep_im_for_blob(img[coor[0]:coor[1],coor[2]:coor[3]], cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        im, imscale = blob_utils.prep_im_for_blob(img_, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            # e_t3 = time.time()
            # t_t3= e_t3 - s_t
            # print("时间_创建新roidb：{}".format(t_t3))
        processed_img.append(im)
        im_scales.append(imscale)
        all_roidb.append(roidb_n)
        #--------------------- 无多尺度下开启旋转模块 --------------------------------------

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_img)
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
        # print(i)
        # print("minibatch.py 187  roidb[boxes]{}".format(roidb[i]['boxes']))
        img_path=roidb[i]['image']
        # print("minibatch.py  189  image:------",img_path)
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
        img_ = img.copy()   # 其实在原图做完计算之后可以不用考虑值会改变的事了
        # s_t = time.time()
        # ---------------------尽量不改变原roidb的值
        roidb_n = roidb[i]
        if ROTATION:  # 是否旋转
            rot_random =  random.uniform(0,1)
            if rot_random <= ROT:
                if ROT_RANDOM :
                    angle = random.randint(-ROT_RANDOM_ANGLE,ROT_RANDOM_ANGLE)
                else: 
                    angle = random.randint(1,4)
                    angle = angle * 90.0
                    
                img_ , h_, w_, l, n_w, n_h = rotate_image(img_, angle, l, w_, h_, n_w, n_h, True)     # 裁剪完后的图
                angle = (angle / 180.0 * math.pi)

                roidb_n = creat_rotation_roidb(roidb_n, angle, MASKRCNN_SWITCH)

            else: pass
        # e_t_ = time.time()
        # t_t_ = e_t_ - s_t
        # print("旋转时间_：{}".format(t_t_))

        coor, new_boxes_crop, img_crop_w, img_crop_h, index_list =  crop(h,w,roidb_n,WIDTH_REGION ,HEIGHT_REGION)
        # 裁剪后实例分割的相对应的segms -------------Maskrcnn的点的扩增模块-----------------------------------
        if MASKRCNN_SWITCH : 
            roidb_n['segms'] = roidb_n['segms'][index_list]
            segms = []
            seg_areas = []
            for m_s, roi_s in enumerate(roidb_n['segms']):
                roi_s = np.array(roi_s).reshape(1,-1,2)
                img_se = np.zeros((roidb_n['height'], roidb_n['width'], 3))
                imag = cv2.drawContours(img_se, roi_s,-1,(255,255,255),-1)
                imag = imag[coor[0]:coor[1], coor[2]:coor[3]]
                imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
                _ ,contours,hierarchy = cv2.findContours(imag,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                for kk, con_point in enumerate(contours):
                    if con_point[0][0] == 0:
                        con_point[0][0] = 1
                    elif con_point[0][0] == img_crop_w -1:
                        con_point[0][0] = con_point[0][0] - 1
                    else: pass
                    if con_point[0][1] == 0:
                        con_point[0][1] = 1
                    elif con_point[0][1] == img_crop_w -1:
                        con_point[0][1] = con_point[0][0] - 1
                    else: pass
                roi_s = contours.reshape(1,-1)
                eara = float(int(cv2.contourArea(contours[0])))
                segms.append(roi_s)
                seg_areas.append(eara)
            
            roidb_n['segms'] = segms
            roidb_n['seg_areas'] = np.array(eara, dtype = np.float)
        
        # -------------------------------------------------------------------------------------------------------
        #  深拷贝以避免改变原始图像数据
        img_ = copy.deepcopy(img_[coor[0]:coor[1],coor[2]:coor[3]])
        # e_t = time.time()
        # t_t = e_t - s_t
        # print("时间：{}".format(t_t))
        # 随机决定要不要做色度上的数据扩增
        if MULTI_COLOR:
            color_random =  random.uniform(0,1)
            #print("ccccccc",color_random)
            if color_random <= GAUSSIAN:
                #print("高斯")
                img_ =  randomGaussian(img_, GAUSSIAN_KERNEL, GAUSSIAN_SIGMAX)
            elif color_random <= CAB:
                pass
            else:
                #print("融合暗")
                img_ = Contrast_and_Brightness(ALPHA, GAMMA, img_)

        if not new_boxes_crop  == []:
            #----如果裁剪完后有缺陷
            #print("执行了minibatch.py 230")
            new_roidb = creat_new_roidb(roidb_n, img_crop_w, img_crop_h, new_boxes_crop, index_list)
            # e_t2 = time.time()
            # t_t2= e_t2 - s_t
            #print("时间_创建新roidb：{}".format(t_t2))
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            #im, imscale = blob_utils.prep_im_for_blob(img[coor[0]:coor[1],coor[2]:coor[3]], cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            im, imscale = blob_utils.prep_im_for_blob(img_, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            # e_t3 = time.time()
            # t_t3= e_t3 - s_t
            # print("时间_创建新roidb：{}".format(t_t3))
            processed_img.append(im)
            imgs_scale.append(imscale)
            all_roidb.append(new_roidb)
        else:
            #----如果裁剪完后是一张好的图
            #print("执行了minibatch.py 239")
            roidb_copy = creat_new_roidb_empty(roidb_n, img_crop_w, img_crop_h)

            target_size=cfg.TRAIN.SCALES[scale_inds[i]]
            # im, imscale = blob_utils.prep_im_for_blob(img[coor[0]:coor[1],coor[2]:coor[3]], cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            im, imscale = blob_utils.prep_im_for_blob(img_, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            processed_img.append(im)
            imgs_scale.append(imscale)
            all_roidb.append(roidb_copy)

    blob=blob_utils.im_list_to_blob(processed_img) 
        # print(" minibatch  266  imgs_scale{}, roidb[boxes]{}]".format(imgs_scale,all_roidb))
    #print(len(all_roidb))
    #print(all_roidb)
    # e_t__ = time.time()
    # t_t__ = e_t__ - s_t
    # print("时间_最终：{}".format(t_t__))
    return blob, imgs_scale, all_roidb

