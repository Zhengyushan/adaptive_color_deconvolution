import numpy as np
import tensorflow as tf
from acd import acd_model


class StainNormalizer(object):
    def __init__(self, pixel_number=100000, step=300, batch_size=1500):
        self._pn = pixel_number
        self._bs = batch_size
        self._step_per_epoch = int(pixel_number / batch_size)
        self._epoch = int(step / self._step_per_epoch)
        self._template_dc_mat = None
        self._template_w_mat = None

    def fit(self, images):
        opt_cd_mat, opt_w_mat = self.extract_adaptive_cd_params(images)
        self._template_dc_mat = opt_cd_mat
        self._template_w_mat = opt_w_mat

    def transform(self, images):
        if self._template_dc_mat is None:
            raise AssertionError('Run fit function first')

        opt_cd_mat, opt_w_mat = self.extract_adaptive_cd_params(images)
        transform_mat = np.matmul(opt_cd_mat * opt_w_mat,
                                  np.linalg.inv(self._template_dc_mat * self._template_w_mat))

        od = -np.log((np.asarray(images, np.float) + 1) / 256.0)
        normed_od = np.matmul(od, transform_mat)
        normed_images = np.exp(-normed_od) * 256 - 1

        return np.maximum(np.minimum(normed_images, 255), 0)

    def he_decomposition(self, images, od_output=True):
        if self._template_dc_mat is None:
            raise AssertionError('Run fit function first')

        opt_cd_mat, _ = self.extract_adaptive_cd_params(images)

        od = -np.log((np.asarray(images, np.float) + 1) / 256.0)
        normed_od = np.matmul(od, opt_cd_mat)

        if od_output:
            return normed_od
        else:
            normed_images = np.exp(-normed_od) * 256 - 1
            return np.maximum(np.minimum(normed_images, 255), 0)


    def sampling_data(self, images):
        pixels = np.reshape(images, (-1, 3))
        pixels = pixels[np.random.choice(pixels.shape[0], min(self._pn * 20, pixels.shape[0]))]
        od = -np.log((np.asarray(pixels, np.float) + 1) / 256.0)
        
        # filter the background pixels (white or black)
        tmp = np.mean(od, axis=1)
        od = od[(tmp > 0.3) & (tmp < -np.log(30 / 256))]
        od = od[np.random.choice(od.shape[0], min(self._pn, od.shape[0]))]

        return od

    def extract_adaptive_cd_params(self, images):
        """
        :param images: RGB uint8 format in shape of [k, m, n, 3], where
                       k is the number of ROIs sampled from a WSI, [m, n] is 
                       the size of ROI.
        """
        od_data = self.sampling_data(images)
        input_od = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        target, cd, w = acd_model(input_od)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for ep in range(self._epoch):
                for step in range(self._step_per_epoch):
                    sess.run(target, {input_od: od_data[step * self._bs:(step + 1) * self._bs]})
            opt_cd = sess.run(cd)
            opt_w = sess.run(w)
        return opt_cd, opt_w
