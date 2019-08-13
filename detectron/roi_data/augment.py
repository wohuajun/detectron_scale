#! /usr/bin/env python
# -*- coding:UTF-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import cv2 as cv
import numpy as np 
import random
import copy
import time
from rotation import *

# 通过尺度随机区间因子,获得裁剪图片的起始点和长宽
def random_corp(width, height, width_random_region, height_random_region):
    # 原图的长宽
    image_width = width
    image_height = height
    # 生成随机因子
    # ------锁定为固定裁剪尺度-----------
    width_random = width_random_region
    height_random = height_random_region
    # ------锁定为固定裁剪尺度-----------
    # width_random = random.uniform(width_random_region[0], width_random_region[1])
    # height_random = random.uniform(height_random_region[0], height_random_region[1])
    # height_random = width_random # 长宽尺度一致 是否需要高级设置
    # 待裁剪框的高和宽的大小
    new_width = int(image_width*width_random)
    new_height = int(image_height*height_random)
    # 获取起始点坐标,在(原图长宽减去待裁剪框高宽后的差,就是可让起始点随机的)范围中随机出坐标
    x_start = random.randint(0,(image_width-new_width))
    y_start = random.randint(0,(image_height-new_height))    
    
    return new_width, new_height, x_start, y_start, width_random ,height_random

# 计算两框的重叠面积,此处该函数有相性,后四组为boxes相关数据
def area_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    
    # 不重叠的情况
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    # 重叠框的宽是两框最后点的x值中的小的减去两框最左边点的值x中的较大的那一个 
    colInt = abs(min(x1 + w1 ,x2 + w2) - max(x1, x2))
    # 重叠框的高是两框最下点的y值中的小的减去两框最上边点的值y中的较大的那一个 
    rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    # 计算重叠区域的面积和两框面积
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    
    if area2 == 0 : 
        return 0

    else:
        return overlap_area / area2 


def crop(h ,w,roidb_data, width_region = 0.5, height_region = 0.5, area_threshold = 0.5):
    # 通过roidb读取图片 并获取图片的高和宽的数据
    roidb = roidb_data
    # 用copy,因为原图的img数据本身也需要被训练
    image_src_h = h
    image_src_w = w
    # 设置裁剪高宽的比例随机数
    w_region, h_region = width_region , height_region
    # 获得裁剪候选框的实际高和宽,以及在原图中该框的起始左上角点的坐标,还有相应的随机比例数(无用)
    img_crop_w, img_crop_h, x_start_coor, y_start_coor, w_scale_random, h_scale_random = random_corp( image_src_w, image_src_h,w_region,h_region)
    # 通过起始点左上角的坐标和高宽,得到裁剪备选框的右下角坐标
    x_end_coor = x_start_coor + img_crop_w
    y_end_coor = y_start_coor + img_crop_h
    # 获取roidb中的boxes的数据,是num_boxes*4的numpy数据
    boxes_list = list(roidb['boxes'])
    # 创建空list以存放裁剪后的boxes数据
    new_boxes_crop = []
    # 最后保留的框boxes的对应的index
    index_list = []

    for i in range(len(boxes_list)):
        # 获取boxes的值
        x_src_min = boxes_list[ i][0]
        y_src_min = boxes_list[ i][1]
        x_src_max = boxes_list[ i][2]
        y_src_max = boxes_list[ i][3]
        # 获得该boxes的高和宽
        box_w = x_src_max - x_src_min
        box_h = y_src_max - y_src_min
        # 获得boxes和待裁剪框的重叠部分面积
        area_overlap_value = area_overlap(x_start_coor, y_start_coor, img_crop_w, img_crop_h, x_src_min, y_src_min, box_w, box_h)
        # 如果速度不理想可以按照坐标先排除不重叠的框和完全包含的框,只计算部分重叠部分,进行计算并裁剪box
        # 如果重叠小于设定的阈值,不将改框添加至新的new_boxes_crop的list
        # if area_overlap_value < area_threshold:
        if area_overlap_value < (area_threshold*w_region):
            pass
        # 如果重叠部分为全部包含,不将改框添加至新的new_boxes_crop的list
        # elif area_overlap_value == 1:
        #     x_min = max(x_src_min, x_start_coor+1)
        #     y_min = max(y_src_min, y_start_coor+1)
        #     x_max = min(x_src_max, x_end_coor-1)
        #     y_max = min(y_src_max, y_end_coor-1)
        #     x_min = x_min - x_start_coor
        #     y_min = y_min - y_start_coor
        #     x_max = x_max - x_start_coor
        #     y_max = y_max - y_start_coor
        #     new_boxes_crop .append( [x_min, y_min, x_max, y_max])
        #     new_classes_crop.append(roidb['gt_classes'][i])
        #     index_list.append(i)

        else:
            x_min = max(x_src_min, x_start_coor+1)
            y_min = max(y_src_min, y_start_coor+1)
            x_max = min(x_src_max, x_end_coor-1)
            y_max = min(y_src_max, y_end_coor-1)
            x_min = x_min - x_start_coor
            y_min = y_min - y_start_coor
            x_max = x_max - x_start_coor
            y_max = y_max - y_start_coor
            new_boxes_crop .append( [x_min, y_min, x_max, y_max])
            index_list.append(i)

            


    return [y_start_coor , y_end_coor, x_start_coor , x_end_coor], new_boxes_crop, img_crop_w, img_crop_h, index_list

#--------色彩相关模块-------------
def randomGaussian(image, gauss_kernel, sigmax):
    temp = cv.GaussianBlur(image, (gauss_kernel, gauss_kernel), sigmax)
    return temp

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype) #dst = alpha * img + beta *blank
    dst = cv.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def random_Gaussian_Brightness():
    pass

# 整体像素乘以系数
def bright_trans(image, threshold):                                                

    image = image * threshold

    return image

# 改变HSV的第三通道的亮度系数
def img2darker(image, thredshold):                                                 
    darker_hsv = image.copy()
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    darker_hsv[:,:,2] = thredshold * img_hsv[:,:,2]
    darker_img = cv.cvtColor(darker_hsv, cv.COLOR_HSV2BGR)
    return darker_img

# Gamma变换是对输入图像灰度值进行的非线性操作
def gamma_trans(img, gamma):                                                        
    gamma_table = [np.power(x/255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    img_corrected = cv.LUT(img, gamma_table)
    return img_corrected
   
#----------------------------

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

def creat_new_roidb(roidb, img_crop_w, img_crop_h, new_boxes_crop, indexlist, coor, MASKRCNN_SWITCH):
    # s_t = time.time()
    #roidb_copy =  copy.deepcopy(roidb)
    roidb_copy = {}
    roidb_copy['boxes'] = np.array(new_boxes_crop, dtype=np.float32)
    roidb_copy['width'] = img_crop_w
    roidb_copy['height'] = img_crop_h
    roidb_copy['has_visible_keypoints'] = roidb['has_visible_keypoints']
    roidb_copy['flipped'] = roidb['flipped']
    roidb_copy['seg_areas'] = roidb['seg_areas'][indexlist]
    roidb_copy['dataset'] = roidb['dataset'] 
    roidb_copy['segms'] = np.array(roidb['segms'])[indexlist]
    roidb_copy['id'] = roidb['id']
    roidb_copy['image'] = roidb['image']
    roidb_copy['gt_classes'] = roidb['gt_classes'][indexlist]
    roidb_copy['gt_overlaps'] = roidb['gt_overlaps'][indexlist]
    roidb_copy['is_crowd'] = roidb['is_crowd'][indexlist]
    roidb_copy['box_to_gt_ind_map'] = roidb['box_to_gt_ind_map'][indexlist]
    roidb_copy['max_classes'] = roidb['max_classes'][indexlist]
    roidb_copy['max_overlaps'] = roidb['max_overlaps'][indexlist]
    box_target=compute_bbox_regression_targets(roidb_copy)
    roidb_copy['bbox_targets']=box_target

    if MASKRCNN_SWITCH : 
        segms = []
        seg_areas = []
        #print("roidb:---------------------",new_roidb )
        if len(roidb_copy['segms'])>0:
            for m_s, roi_s in enumerate(roidb_copy['segms']):
                roi_s = np.array(roi_s[0][:-4]).reshape(1,-1,2)
                img_se = np.zeros((roidb['height'], roidb['width'], 3))
                img_se_ = img_se.copy()[coor[0]:coor[1], coor[2]:coor[3]]
                imag = cv.drawContours(img_se, [roi_s],-1,(255,255,255),-1)
                #cv.imwrite("mask_raw.jpg", imag)
                imag = imag[coor[0]:coor[1], coor[2]:coor[3]]
                imag =  np.array(imag, dtype = np.uint8)
                gray = cv.cvtColor(imag, cv.COLOR_BGR2GRAY)  
                #  一定要做二值化,否则出来的是散点
                ret, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)  
                _, contours, hierarchy = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
                img_fill = cv.drawContours(imag, contours,-1,(0,255,0),-1)
                if len(contours) > 0:
                    for kk, con_point in enumerate(contours):
                        if con_point[0][0][0] == 0:
                            con_point[0][0][0] = 1
                        elif con_point[0][0][0] == img_crop_w -1:
                            con_point[0][0][0] = img_crop_w - 2
                        else: pass
                        if con_point[0][0][1] == 0:
                            con_point[0][0][1] = 1
                        elif con_point[0][0][1] == img_crop_h -1:
                            con_point[0][0][1] = img_crop_h - 2
                        else: pass
                    roi_s = np.array(contours).reshape(1,-1)
                    area = float(int(cv.contourArea(contours[0])))
                    segms.append(roi_s.tolist())
                    seg_areas.append(area)
 
            roidb_copy['segms'] = segms
            roidb_copy['seg_areas'] = np.array(seg_areas, dtype = np.float32)
    return roidb_copy


def creat_new_roidb_empty(roidb, img_crop_w, img_crop_h):
    s_t = time.time()
    #roidb_copy =  copy.deepcopy(roidb)
    roidb_copy = {}
    # e_t1 = time.time()
    # t_t1 = e_t1 - s_t
    # print("计算max_classes 181  耗时{}".format(t_t1))
    roidb_copy['boxes'] = roidb['boxes'][:0]   # 空
    # e_t2 = time.time()
    # t_t2 = e_t2 - s_t
    # print("计算max_classes 185  耗时{}".format(t_t2))
    roidb_copy['width'] = img_crop_w
    roidb_copy['height'] = img_crop_h
    roidb_copy['has_visible_keypoints'] = roidb['has_visible_keypoints']
    roidb_copy['flipped'] = roidb['flipped']
    roidb_copy['seg_areas'] = roidb['seg_areas'][:0]  # 空
    roidb_copy['dataset'] = roidb['dataset']
    roidb_copy['segms'] = roidb['segms'][:0]   # 空
    roidb_copy['id'] = roidb['id']
    roidb_copy['image'] = roidb['image']
    roidb_copy['gt_classes'] = roidb['gt_classes'][:0]
    roidb_copy['gt_overlaps'] = roidb['gt_overlaps'][:0]
    roidb_copy['is_crowd'] = roidb['is_crowd'][:0]
    roidb_copy['box_to_gt_ind_map'] = roidb['box_to_gt_ind_map'][:0]
    roidb_copy['max_classes'] = roidb['max_classes'][:0]
    # e_t3 = time.time()
    # t_t3 = e_t3 - s_t
    # print("计算max_classes 189  耗时{}".format(t_t3))
    roidb_copy['max_overlaps'] = roidb['max_overlaps'][:0]
    #e_t4 = time.time()
    #t_t4 = e_t4 - s_t
    # print("计算box_target前 190  耗时{}".format(t_t4))
    box_target=compute_bbox_regression_targets(roidb_copy)
    roidb_copy['bbox_targets']=box_target
    #e_t = time.time()
    #t_t = e_t - s_t
    # print("创建新的roidb   augment 193  耗时{}".format(t_t))

    return roidb_copy


def creat_rotation_roidb(roidb, angle, l, w_, h_, n_w, n_h, MASKRCNN_SWITCH = False):
        roidb_n = {}
        roidb_n['boxes'] = copy.deepcopy(roidb['boxes'])
        roidb_n['width'] =  n_w
        roidb_n['height'] = n_h
        roidb_n['has_visible_keypoints'] = roidb['has_visible_keypoints']
        roidb_n['flipped'] = roidb['flipped']
        roidb_n['seg_areas'] = copy.deepcopy(roidb['seg_areas'])
        roidb_n['dataset'] = roidb['dataset']
        roidb_n['segms'] = copy.deepcopy(roidb['segms'])
        roidb_n['id'] = roidb['id']
        roidb_n['image'] = roidb['image']
        roidb_n['gt_classes'] = roidb['gt_classes']
        roidb_n['gt_overlaps'] = roidb['gt_overlaps']
        roidb_n['is_crowd'] = roidb['is_crowd']
        roidb_n['box_to_gt_ind_map'] = roidb['box_to_gt_ind_map']
        roidb_n['max_classes'] = roidb['max_classes']
        roidb_n['max_overlaps'] = roidb['max_overlaps']
        roidb_n['bbox_targets'] = roidb['bbox_targets']
        for m, roi_boxes in enumerate(roidb_n['boxes']):
            roi_boxes[0] = roi_boxes[0] + int((l-w_)/2.0)
            roi_boxes[1] = roi_boxes[1] + int((l-h_)/2.0)
            roi_boxes[2] = roi_boxes[2] + int((l-w_)/2.0)
            roi_boxes[3] = roi_boxes[3] + int((l-h_)/2.0)
            x0, y0 = rot_compute(l/2, l/2, roi_boxes[0], roi_boxes[1], angle)
            x1, y1 = rot_compute(l/2, l/2, roi_boxes[2], roi_boxes[1], angle)
            x2, y2 = rot_compute(l/2, l/2, roi_boxes[0], roi_boxes[3], angle)
            x3, y3 = rot_compute(l/2, l/2, roi_boxes[2], roi_boxes[3], angle)
            min_x, min_y, max_x, max_y = box_coordinate(x0,y0,x1,y1,x2,y2,x3,y3)
            min_x = min_x - int((l-n_w)/2.0)
            min_y = min_y - int((l-n_h)/2.0)
            max_x = max_x - int((l-n_w)/2.0)
            max_y = max_y - int((l-n_h)/2.0)
            roidb_n['boxes'][m][0] = int(min_x) 
            roidb_n['boxes'][m][1] = int(min_y) 
            roidb_n['boxes'][m][2] = int(max_x) 
            roidb_n['boxes'][m][3] = int(max_y) 
        # Maskrcnn的点的扩增模块----------------------------------------------------------------
        if MASKRCNN_SWITCH : 
            for m_segme, roi_segms in enumerate(roidb_n['segms']):
                for m_segme_index in range(int(len(roi_segms)/2)):
                    roi_segms[m_segme][m_segme_index] = roi_segms_point + int((l-w_)/2.0)
                    roi_segms[m_segme][m_segme_index + 1] = roi_segms_point + int((l-h_)/2.0)
                    x_segme, y_segme = rot_compute(l/2, l/2, roi_segms[m_segme][m_segme_index], roi_segms[m_segme][m_segme_index + 1], angle)
                    roidb_n['segms'][m_segme][m_segme_index] = int(x_segme) 
                    roidb_n['segms'][m_segme][m_segme_index + 1] = int(y_segme) 
        # -----------------------------------------------------------------------------------

        return roidb_n
