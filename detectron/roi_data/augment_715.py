#! /usr/bin/env python
# -*- coding:UTF-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np 
import random

# 通过尺度随机区间因子,获得裁剪图片的起始点和长宽
def random_corp(image_src_data, width, height, width_random_region, height_random_region):
    # 原图的长宽
    image_width = width
    image_height = height
    # 生成随机因子
    width_random = random.uniform(width_random_region[0], width_random_region[1])
    height_random = random.uniform(height_random_region[0], height_random_region[1])
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


def crop(img, roidb_data, width_region = [0.5, 0.9], height_region = [0.5, 0.9], area_threshold = 0.8):
    # 通过roidb读取图片 并获取图片的高和宽的数据
    roidb = roidb_data
    # 用copy,因为原图的img数据本身也需要被训练
    image = img.copy()
    image_src_h ,image_src_w = image.shape[:2]
    # 设置裁剪高宽的比例随机数
    w_region, h_region = width_region , height_region
    # 获得裁剪候选框的实际高和宽,以及在原图中该框的起始左上角点的坐标,还有相应的随机比例数(无用)
    img_crop_w, img_crop_h, x_start_coor, y_start_coor, w_scale_random, h_scale_random = random_corp(image, image_src_w, image_src_h,w_region,h_region) 
    # 通过起始点左上角的坐标和高宽,得到裁剪备选框的右下角坐标
    x_end_coor = x_start_coor + img_crop_w
    y_end_coor = y_start_coor + img_crop_h
    # 获取roidb中的boxes的数据,是num_boxes*4的numpy数据
    boxes_list = list(roidb['boxes'])
    # 创建空list以存放裁剪后的boxes数据
    new_boxes_crop = []
    new_classes_crop = []
    # 最后保留的框boxes的对应的index
    index_list = []

    for i in range(len(boxes_list)//4):
        # 获取boxes的值
        x_src_min = boxes_list[(0 + i*4)]
        y_src_min = boxes_list[(1 + i*4)]
        x_src_max = boxes_list[(2 + i*4)]
        y_src_max = boxes_list[(3 + i*4)]
        # 获得该boxes的高和宽
        box_w = x_src_max - x_src_min
        box_h = y_src_max - y_src_min
        # 获得boxes和待裁剪框的重叠部分面积
        area_overlap_value = area_overlap(x_start_coor, y_start_coor, img_crop_w, img_crop_h, x_src_min, y_src_min, box_w, box_h)
        # 如果速度不理想可以按照坐标先排除不重叠的框和完全包含的框,只计算部分重叠部分,进行计算并裁剪box
        # 如果重叠小于设定的阈值,不将改框添加至新的new_boxes_crop的list
        if area_overlap_value < area_threshold:
            pass
        # 如果重叠部分为全部包含,不将改框添加至新的new_boxes_crop的list
        # elif area_overlap_value == 1:
        #     x_min = x_src_min - x_start_coor
        #     y_min = y_src_min - y_start_coor
        #     x_max = x_src_max - x_start_coor
        #     y_max = y_src_max - y_start_coor
        #     new_boxes_crop += [x_min, y_min, x_max, y_max]
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
            new_boxes_crop += [x_min, y_min, x_max, y_max]
            new_classes_crop.append(roidb['gt_classes'][i])
            index_list.append(i)

    image = image[y_start_coor : y_end_coor, x_start_coor : x_end_coor]

    return image, new_boxes_crop, new_classes_crop, img_crop_w, img_crop_h, index_list


def randomGaussian(image, gauss_kernel, sigmax):
    temp = cv2.GaussianBlur(image, (gauss_kernel, gauss_kernel), sigmax)
    return temp

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype) #dst = alpha * img + beta *blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


def random_Gaussian_Brightness()