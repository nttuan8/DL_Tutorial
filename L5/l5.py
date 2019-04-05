# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:00:00 2019

@author: DELL
"""

import cv2

img = cv2.imread("gray.jpg", 0)

gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

cv2.imshow('gray.jpg', gray)