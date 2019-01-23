"""
An example for histological images color normalization based on the adaptive color deconvolution as described in the paper:
https://github.com/Zhengyushan/adaptive_color_deconvolution

Yushan Zheng, Zhiguo Jiang, Haopeng Zhang, Fengying Xie, Jun Shi, and Chenghai Xue.
Adaptive Color Deconvolution for Histological WSI Normalization.
Computer Methods and Programs in Biomedicine, v170C (2019) pp.107-120.

"""

import os
import time

import cv2
import numpy as np

from stain_normalizer import StainNormalizer

# disable GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

source_image_dir = 'data/images'
template_dir = 'data/template'
result_dir = 'data/result'

# load template images
template_list = os.listdir(template_dir)
temp_images = np.zeros((template_list.__len__(), 2048, 2048, 3), np.uint8)
for i, name in enumerate(template_list):
    temp_images[i] = cv2.imread(os.path.join(template_dir, name))

# fit
st = time.time()
normalizer = StainNormalizer()
normalizer.fit(temp_images)
print('fit time', time.time() - st)

# normalization
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

slide_list = os.listdir(source_image_dir)
for s in slide_list:
    print('normalize slide', s)
    slide_dir = os.path.join(source_image_dir, s)
    image_list = os.listdir(slide_dir)
    images = np.zeros((image_list.__len__(), 2048, 2048, 3), np.uint8)
    for i, name in enumerate(image_list):
        images[i] = cv2.imread(os.path.join(slide_dir, name))

    st = time.time()
    results = normalizer.transform(images)
    print('transform time', time.time() - st)
    for i, result in enumerate(results):
        cv2.imwrite(os.path.join(result_dir, s) + '_{}_origin.jpg'.format(i), images[i])
        cv2.imwrite(os.path.join(result_dir, s) + '_{}_norm.jpg'.format(i), result)
