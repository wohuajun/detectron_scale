# -*- coding:UTF-8 -*-
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
import scipy.sparse

from detectron.core.config import cfg
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.retinanet as retinanet_roi_data
import detectron.utils.boxes as box_utils
import detectron.roi_data.rpn as rpn_roi_data
import detectron.utils.blob as blob_utils
import random

WIDTH = 1280
HEIGHT = 960
REAL_CLASS = 9
logger = logging.getLogger(__name__)
BRIGHTNESS_CONTRAST = 1
WIDTH_FIX = 2048

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
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
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
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

def get_minibatch_s6(roidb,roidb_noclass):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    if 0:
        random_bbox = dict()
        random_bbox['kernel_size'] = 224
        random_bbox['tl_x'] = random.randint(0, 800)
        random_bbox['tl_y'] = random.randint(0, 800)
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    im_blob, im_scales,error_flag = _get_image_blob_s6(roidb,roidb_noclass)
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




def randomGaussian(image,gauss_kernel,sigmax):
    temp = cv2.GaussianBlur(image, (gauss_kernel, gauss_kernel), sigmax)
    return temp

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype) #dst = alpha * img + beta *blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)                                                  # roidb是一个list数据,每一个元素是dict,获取roidb的数据个数,即样本数
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )                                                                        # 生成每一张对应的scale尺度的list
    processed_ims = []
    im_scales = []
    Gauss_kernel = 0
    SigmaX = 0
    alpha = 1
    beta = 0
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])                                    # 读取对应的数据图片
        if cfg.TRAIN.AUGUMENT:
            im_tmp  = cv2.imread(roidb[i]['image'])
            random_flag = random.randint(0, 1)                                # 随机进行高斯模糊 
            #print(random_flag)
            if random_flag :
                Gauss_kernel = 7#np.random.randint(5, 7)
                SigmaX = 5.5#np.random.randint(0, 70) /10.0

                im = randomGaussian(im_tmp,Gauss_kernel,SigmaX)
                alpha = np.random.randint(0, 15) / 10.0
                beta = np.random.randint(0, 15)
                im = Contrast_and_Brightness(alpha, beta, im)
                #cv2.imwrite('test.jpg',im)
            else:
                im = im_tmp.copy()
        #print(random_flag,alpha,beta)
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        if roidb[i]['flipped']:                                              # 如果flipped为真  则翻转数据
            im = im[:, ::-1, :]                                              # 第二个,前的-1表示对x轴的数据反向读取,即对图进行x轴的翻转
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )                                                                    # 会做一波预处理,resize为宽为600的图
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales


def mat_inter(box1,box2):                                                     # 判断是否框重叠

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

def solve_coincide(box1,box2):                                                # 计算重叠部分占匡二的占比
    # box=(xA,yA,xB,yB)

    if mat_inter(box1,box2)==True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        coincide=intersection/(area2+0.00001)#(area1+area2-intersection)     # 因为area2有可能为0
        return coincide
    else:
        return False
def compute_bbox_regression_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

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


def _get_image_blob_s6(roidb,roidb_noclass1):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """


    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    error_flag = [0,0]

    for i in range(num_images):
        roidb_noclass = roidb_noclass1.copy()
        if roidb[i][u'image'].split('/')[-1]==u'test.jpg': #test.jpg
            random_bbox = dict()
            random_bbox['kernel_size_x'] = int(WIDTH / 5)
            random_bbox['kernel_size_y'] = int(HEIGHT / 5)
            random_bbox['tl_x'] = 0
            random_bbox['tl_y'] = 0
            x0 = random_bbox['tl_x']
            x1 = random_bbox['tl_x'] + random_bbox['kernel_size_x']
            y0 = random_bbox['tl_y']
            y1 = random_bbox['tl_y'] + random_bbox['kernel_size_y']
            im = cv2.imread(roidb[i]['image'])[y0:y1, x0:x1]
            im = cv2.resize(im, (WIDTH, HEIGHT))
            # cv2.imwrite('/home/icubic/aa.png',im)
            error_flag[i] = 0
            roidb[i] = roidb_noclass.copy()
            roidb[i][u'height'] = HEIGHT
            roidb[i][u'width'] = WIDTH
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = blob_utils.prep_im_for_blob(
                im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            )
            im_scales.append(im_scale)
            processed_ims.append(im)
        else:
            if 1:
                if len(roidb[i][u'boxes']) == 0:
                    #print('no bbox')
                    random_bbox = dict()
                    random_flag = random.randint(0, 1)
                    real_yuanlai_width = roidb[i][u'width'] * 1
                    real_yuanlai_height = roidb[i][u'height'] * 1
                    width_ratio = float(real_yuanlai_width) / WIDTH_FIX#1024
                    height_after_ratio = int(float(real_yuanlai_height) / width_ratio)
                    width_after_ratio =WIDTH_FIX #1024
                    if 1:
                        if random_flag == 0:
                            #print(random_flag)
                            random_bbox['kernel_size_x'] = int(float(width_after_ratio) / 5.0)#int(WIDTH / 5.0)
                            random_bbox['kernel_size_y'] = int(float(height_after_ratio) / 5.0)#int(HEIGHT / 5.0)

                            random_X = width_after_ratio - random_bbox['kernel_size_x']
                            random_Y = height_after_ratio - random_bbox['kernel_size_y']
                            try:
                                random_bbox['tl_x'] = random.randint(0, random_X)
                                random_bbox['tl_y'] = random.randint(0, random_Y)
                            except:
                                print('aa')
                            x0 = random_bbox['tl_x']
                            x1 = random_bbox['tl_x'] + random_bbox['kernel_size_x']
                            y0 = random_bbox['tl_y']
                            y1 = random_bbox['tl_y'] + random_bbox['kernel_size_y']
                            im = cv2.imread(roidb[i][u'image'])
                            im = cv2.resize(im, (width_after_ratio, height_after_ratio))[y0:y1, x0:x1]
                            im = cv2.resize(im, (WIDTH, HEIGHT))
                            roidb[i] = roidb_noclass.copy()
                            roidb[i][u'height'] = HEIGHT
                            roidb[i][u'width'] = WIDTH
                        else:
                #print(random_flag)
                            random_bbox['kernel_size_x'] = int(float(width_after_ratio) / 1.0)
                            random_bbox['kernel_size_y'] = int(float(height_after_ratio) / 1.0)

                            random_X = width_after_ratio - random_bbox['kernel_size_x']
                            random_Y = height_after_ratio - random_bbox['kernel_size_y']
                            random_bbox['tl_x'] = random.randint(0, random_X)
                            random_bbox['tl_y'] = random.randint(0, random_Y)
                            x0 = random_bbox['tl_x']
                            x1 = random_bbox['tl_x'] + random_bbox['kernel_size_x']
                            y0 = random_bbox['tl_y']
                            y1 = random_bbox['tl_y'] + random_bbox['kernel_size_y']
                            im = cv2.imread(roidb[i][u'image'])
                            im = cv2.resize(im, (width_after_ratio, height_after_ratio))[y0:y1, x0:x1]
                            im = cv2.resize(im, (WIDTH, HEIGHT))
                            roidb[i] = roidb_noclass.copy()
                            roidb[i][u'height'] = HEIGHT
                            roidb[i][u'width'] = WIDTH
                    else:
                        im = cv2.imread(roidb[i][u'image'])
                        im = cv2.resize(im, (WIDTH, HEIGHT))
                        roidb[i] = roidb_noclass.copy()
                        roidb[i][u'height'] = HEIGHT
                        roidb[i][u'width'] = WIDTH
                    # cv2.imwrite('/home/icubic/daily_work/circruit_model/tmp_images/aa.png',im)
                    assert im is not None, \
                        'Failed to read image \'{}\''.format(roidb[i]['image'])
                    if roidb[i]['flipped']:#for image flip background training
                        im = im[:, ::-1, :]
                    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
                    im, im_scale = blob_utils.prep_im_for_blob(
                        im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
                    )
                    im_scales.append(im_scale)
                    processed_ims.append(im)
                    continue
                #if len(roidb[i][u'boxes']==1):
                #    print('roidb=1')
                #if len(roidb[i][u'boxes']>1):
                #    print('roidb>1')
                real_yuanlai_width = roidb[i][u'width'] * 1
                real_yuanlai_height = roidb[i][u'height'] * 1
                width_ratio = float(real_yuanlai_width) / WIDTH_FIX
                height_after_ratio = int(float(real_yuanlai_height) / width_ratio)
                width_after_ratio = WIDTH_FIX

                real_class = []#roidb[i]['gt_classes'][0]
                num_real_class = len(roidb[i]['gt_classes'])

                random_bbox = dict()
                random_bbox['kernel_size_x'] = int(float(width_after_ratio) / 5.0)#int(WIDTH_FIX / 5)
                random_bbox['kernel_size_y'] = int(float(height_after_ratio) / 5.0)#int(WIDTH_FIX / 5)
                if 1:
                    w_tongji = 0
                    h_tongji = 0
                    for i_tongji, sub_boxes_tongji in enumerate(roidb[i][u'boxes']):
                        crop_x0_tongji = int(sub_boxes_tongji[0] / real_yuanlai_width * width_after_ratio)
                        crop_y0_tongji = int(sub_boxes_tongji[1] / real_yuanlai_height * height_after_ratio)
                        crop_x1_tongji = int(sub_boxes_tongji[2] / real_yuanlai_width * width_after_ratio)
                        crop_y1_tongji = int(sub_boxes_tongji[3] / real_yuanlai_height * height_after_ratio)
                        w_tongji = crop_x1_tongji - crop_x0_tongji
                        h_tongji = crop_y1_tongji - crop_y0_tongji
                        if w_tongji>int(WIDTH_FIX / 5.0) or h_tongji>int(WIDTH_FIX / 5.0):
                            random_bbox['kernel_size_x'] = int(float(width_after_ratio) / 1.0)
                            random_bbox['kernel_size_y'] = int(float(height_after_ratio) / 1.0)
                            break

                if random_bbox['kernel_size_x']==int(WIDTH_FIX / 5.0):
                    random_X = width_after_ratio - random_bbox['kernel_size_x']
                    random_Y = height_after_ratio - random_bbox['kernel_size_y']
                    random_bbox['tl_x'] = random.randint(0, random_X)
                    random_bbox['tl_y'] = random.randint(0, random_Y)
                    x0 = random_bbox['tl_x']
                    x1 = random_bbox['tl_x'] + random_bbox['kernel_size_x']
                    y0 = random_bbox['tl_y']
                    y1 = random_bbox['tl_y'] + random_bbox['kernel_size_y']
                    try:
                        im = cv2.imread(roidb[i][u'image'])
                    except:
                        im = cv2.imread(roidb[i][u'image'])
                    im = cv2.resize(im, (width_after_ratio, height_after_ratio))[y0:y1, x0:x1]
                    im = cv2.resize(im, (WIDTH, HEIGHT))
                    sum_inside_overlaps = 0
                    boxes_inside_overlaps = []

                    for i_roidb,sub_boxes in enumerate(roidb[i][u'boxes']):
                        crop_x0 = int(sub_boxes[0]/real_yuanlai_width*width_after_ratio)
                        crop_y0 = int(sub_boxes[1]/real_yuanlai_height*height_after_ratio)
                        crop_x1 = int(sub_boxes[2]/real_yuanlai_width*width_after_ratio)
                        crop_y1 = int(sub_boxes[3]/real_yuanlai_height*height_after_ratio)
                        #real_x0 = float(crop_x0 - x0)*1024/224  # float(crop_x0) / 1024 * 224
                        #real_y0 = float(crop_y0 - y0)*1024/224  # float(crop_y0) / 1024 * 224
                        #real_x1 = float(crop_x1 - x0)*1024/224  # float(crop_x1) / 1024 * 224
                        #real_y1 = float(crop_y1 - y0)*1024/224


                        overlaps_rate = solve_coincide((x0, y0, x1, y1), (crop_x0, crop_y0, crop_x1, crop_y1))
                        if overlaps_rate>0.9:
                            sum_inside_overlaps = sum_inside_overlaps + 1
                            #real_x0 = crop_x0 - x0  # float(crop_x0) / 1024 * 224
                            #real_y0 = crop_y0 - y0  # float(crop_y0) / 1024 * 224
                            #real_x1 = crop_x1 - x0  # float(crop_x1) / 1024 * 224
                            #real_y1 = crop_y1 - y0

                            real_x0 = float(crop_x0 - x0)*WIDTH/(random_bbox['kernel_size_x'])  # float(crop_x0) / 1024 * 224
                            real_y0 = float(crop_y0 - y0)*HEIGHT/(random_bbox['kernel_size_y']) # float(crop_y0) / 1024 * 224
                            real_x1 = float(crop_x1 - x0)*WIDTH/(random_bbox['kernel_size_x'])  # float(crop_x1) / 1024 * 224
                            real_y1 = float(crop_y1 - y0)*HEIGHT/(random_bbox['kernel_size_y'])
                            if real_x0<0:
                                real_x0 = 0
                            if real_x0>WIDTH:
                                real_x0 = WIDTH

                            if real_x1<0:
                                real_x1 = 0
                            if real_x1>WIDTH:
                                real_x1 = WIDTH

                            if real_y0<0:
                                real_y0 = 0
                            if real_y0>HEIGHT:
                                real_y0 = HEIGHT

                            if real_y1<0:
                                real_y1 = 0
                            if real_y1>HEIGHT:
                                real_y1 = HEIGHT
                            #cv2.rectangle(im, (int(real_x0), int(real_y0)), (int(real_x1), int(real_y1)), (0, 255, 255), 3)
                            #cv2.imwrite('/home/icubic/daily_work/code/Detectron/detectron/datasets/data/shanghai/aa.png',im)

                            boxes_inside_overlaps.append([real_x0, real_y0, real_x1, real_y1])
                            real_class.append(roidb[i]['gt_classes'][i_roidb])
                            #cv2.rectangle(im, (int(real_x0), int(real_y0)),
                                          #(int(real_x1), int(real_y1)), (255, 0, 255))
                            #cv2.imwrite('/home/icubic/daily_work/code/circruit/new/result/uu.png', im)
                    #a = roidb[i]['gt_overlaps'].toarray()

                    if sum_inside_overlaps>0 :
                        num_valid_objs = sum_inside_overlaps*1
                        boxes = np.zeros((num_valid_objs, 4), dtype=np.float32)
                        gt_classes = np.zeros((num_valid_objs), dtype=np.int32)
                        gt_overlaps = np.zeros((num_valid_objs, REAL_CLASS), dtype=np.float32)
                        box_to_gt_ind_map = np.zeros((num_valid_objs), dtype=np.int32)
                        is_crowd = np.zeros((num_valid_objs), dtype=np.bool)
                        for ix in range(num_valid_objs):
                            gt_classes[ix] = real_class[ix]#real_class*1
                            try:
                                gt_overlaps[ix, real_class] = 1.0
                            except:
                                print('error')
                            is_crowd[ix] = False
                            box_to_gt_ind_map[ix] = ix
                            for i_index in range(4):
                                boxes[ix,i_index] = boxes_inside_overlaps[ix][i_index]

                        #for ix in range(num_valid_objs):
                            #box_to_gt_ind_map[ix] = ix
                            #cls = real_class*1
                        roidb_noclass['boxes'] = np.append(roidb_noclass['boxes'], boxes, axis=0)

                        roidb_noclass['gt_classes'] = np.append(roidb_noclass['gt_classes'], gt_classes)
                        #mm = np.append(
                        #    roidb_noclass['gt_overlaps'].toarray(), gt_overlaps,axis=0)
                        roidb_noclass['gt_overlaps'] = np.append(
                            roidb_noclass['gt_overlaps'].toarray(), gt_overlaps)
                        roidb_noclass['gt_overlaps'] = scipy.sparse.csr_matrix(roidb_noclass['gt_overlaps'])
                        #mm = np.append(mm, gt_overlaps, axis=0)
                        #roidb_noclass['gt_overlaps'] = scipy.sparse.csr_matrix(mm)
                        roidb_noclass['is_crowd'] = np.append(roidb_noclass['is_crowd'], is_crowd)
                        roidb_noclass['box_to_gt_ind_map'] = np.append(roidb_noclass['box_to_gt_ind_map'], box_to_gt_ind_map)

                        gt_overlaps = roidb_noclass['gt_overlaps'].toarray()
                        # max overlap with gt over classes (columns)
                        max_overlaps = gt_overlaps.max(axis=1)
                        # gt class that had the max overlap
                        max_classes = gt_overlaps.argmax(axis=1)
                        roidb_noclass['max_classes'] = max_classes
                        roidb_noclass['max_overlaps'] = max_overlaps
                        # sanity checks
                        # if max overlap is 0, the class must be background (class 0)
                        zero_inds = np.where(max_overlaps == 0)[0]
                        assert all(max_classes[zero_inds] == 0)
                        # if max overlap > 0, the class must be a fg class (not class 0)
                        nonzero_inds = np.where(max_overlaps > 0)[0]
                        assert all(max_classes[nonzero_inds] != 0)
                        roidb_noclass['bbox_targets'] = compute_bbox_regression_targets(roidb_noclass)
                        roidb[i] = roidb_noclass.copy()
                        roidb[i][u'height'] = HEIGHT
                        roidb[i][u'width'] = WIDTH

                    else:
                        roidb[i] = roidb_noclass.copy()
                        roidb[i][u'height'] = HEIGHT
                        roidb[i][u'width'] = WIDTH




            #print('aa')
                    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
                    im, im_scale = blob_utils.prep_im_for_blob(
                        im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
                    )
                    im_scales.append(im_scale)
                    processed_ims.append(im)
                else:
                    im = cv2.imread(roidb[i]['image'])
                    assert im is not None, \
                        'Failed to read image \'{}\''.format(roidb[i]['image'])
                    if roidb[i]['flipped']:
                        im = im[:, ::-1, :]
                    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
                    im, im_scale = blob_utils.prep_im_for_blob(
                        im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
                    )
                    im_scales.append(im_scale)
                    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales,error_flag


