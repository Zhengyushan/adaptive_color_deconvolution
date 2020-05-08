"""
An example for histological images color normalization based on the adaptive color deconvolution as described in the paper:
https://github.com/Zhengyushan/adaptive_color_deconvolution

Yushan Zheng, Zhiguo Jiang, Haopeng Zhang, Fengying Xie, Jun Shi, and Chenghai Xue.
Adaptive Color Deconvolution for Histological WSI Normalization.
Computer Methods and Programs in Biomedicine, v170 (2019) pp.107-120.

"""
import os
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
temp_images = np.asarray([cv2.imread(os.path.join(template_dir, name)) for name in template_list])

# extract the stain parameters of the template slide
normalizer = StainNormalizer()
normalizer.fit(temp_images[:,:,:,[2,1,0]]) #BGR2RGB

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# normalization
slide_list = os.listdir(source_image_dir)
for s in slide_list:
    print('normalize slide', s)
    slide_dir = os.path.join(source_image_dir, s)
    image_list = os.listdir(slide_dir)
    images = np.asarray([cv2.imread(os.path.join(slide_dir, name)) for name in image_list])

    ## color transform
    results = normalizer.transform(images[:,:,:,[2,1,0]]) #BGR2RGB
    # display
    for i, result in enumerate(results):
        cv2.imwrite(os.path.join(result_dir, s + '_{}_origin.jpg'.format(i)), images[i])
        cv2.imwrite(os.path.join(result_dir, s + '_{}_norm.jpg'.format(i)) , result[:,:,[2,1,0]]) #RGB2BGR

    ## h&e decomposition
    he_channels = normalizer.he_decomposition(images[:,:,:,[2,1,0]], od_output=True) #BGR2RGB
    # debug display
    for i, result in enumerate(he_channels):
        cv2.imwrite(os.path.join(result_dir, s + '_{}_h.jpg'.format(i)), result[:,:,0]*128)
        cv2.imwrite(os.path.join(result_dir, s + '_{}_e.jpg'.format(i)), result[:,:,1]*128)

