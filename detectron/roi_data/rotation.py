#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 08:38:19 2017
@modify_author: RAYOU
@Email:whjinc@163.com
"""

from __future__ import print_function
import cv2
import os
import numpy as np
import random
import sys
from math import * 

#　lambda函数，用来扣取图片的部分
crop_image = lambda img, x0, y0, x1, y1: img[y0:y1, x0:x1]


#
def rot_compute(x0, y0, x1, y1, angle_):
    dx = x1 - x0
    dy = y0 - y1
#    print("笛卡尔坐标下的距离为：{0}{1}".format(dx, dy))
    magnitude, angle =	cv2.cartToPolar(dx, dy)
#    print("距离{}角度{}".format(magnitude, angle))
    angle = angle + angle_
    dx_, dy_ = cv2.polarToCart(magnitude, angle, True)
#    print(dx_, dy_)
    x2 = dx_ + x0
    y2 = y0 - dy_ 
    #print(x2[0], y2[0])
    return x2[0], y2[0]


# 通过四个坐标点求出最小外切水平矩形的坐标
def box_coordinate(x0,y0,x1,y1,x2,y2,x3,y3):
    min_x = min(x0, x1, x2, x3)
    min_y = min(y0, y1, y2, y3)
    max_x = max(x0, x1, x2, x3)
    max_y = max(y0, y1, y2, y3)
    
    return int(min_x), int(min_y), int(max_x), int(max_y)

# 此为自己写的极坐标旋转函数　　和opencv自带的有不同　　　如有问题可以改用其他
def coordinate_translate(theta, x, y, c_x, c_y):
   beta = (atan((float(c_y - y) / float(x-c_x)))/pi) *180 
   #print("角度{}".format(beta))
   gamma = theta + beta
   gamma = (gamma/180.0) * pi
   R = sqrt(pow((y-c_y),2)+pow((x-c_x),2))
   delta_x = cos(gamma)*R
   delta_y = sin(gamma)*R
   x_ = c_x + delta_x
   y_ = c_y + delta_y
   
   return x_, y_

# 对图片的旋转
def rotate_image(img, angle, crop):                                                 
    h, w, c = img.shape
    l = int(sqrt(pow(h,2)+pow(w,2))) +1
    src = np.zeros((l,l,c))
    w_s = int((l-w)/2)
    h_s = int((l-h)/2)
    src[h_s:h_s+h, w_s:w_s+w] = img
    # cv2.imwrite("demo2.jpg", src)
    M_rotate = cv2.getRotationMatrix2D((l/2, l/2), angle, 1)	
    img_rotated = cv2.warpAffine(src, M_rotate, (l, l))
    # cv2.imwrite("demo1.jpg", img_rotated)
    nw = l
    nh = l
    
    
    if crop:
        #print(l/2, l/2)
        #print(w_s, h_s)
        x0, y0 = coordinate_translate(angle + 180 , w_s, h_s, l/2, l/2) 
        #print( w_s + w, h_s)
        x1, y1 = coordinate_translate(angle, w_s + w , h_s, l/2, l/2)
        x2, y2 = coordinate_translate(angle +180, w_s, h_s + h, l/2, l/2)
        x3, y3 = coordinate_translate(angle +360, w_s + w, h_s + h, l/2, l/2)
        min_x,min_y, max_x, max_y = box_coordinate(x0,y0,x1,y1,x2,y2,x3,y3)
        #print( x0,y0,x1,y1,x2,y2,x3,y3 )
        #print( min_x,min_y, max_x, max_y )
        img_rotated = crop_image(img_rotated, min_x,min_y, max_x, max_y)
        nw = (max_x - min_x)
        nh = (max_y - min_y)
    return img_rotated, h, w, l , nw , nh
