import numpy as np
import tensorflow as tf

# initial varphi for rgb input
init_varphi = np.asarray([[0.294, 0.110, 0.894],
                          [0.750, 0.088, 0.425]])

# # initial varphi for bgr input
# init_varphi = np.asarray([[0.6060, 1.2680, 0.7989],
#                           [1.2383, 1.2540, 0.3927]])

def acd_model(input_od, lambda_p=0.002, lambda_b=10, lambda_e=1, eta=0.6, gamma=0.5):
    """
    Stain matrix estimation by
    "Yushan Zheng, et al., Adaptive Color Deconvolution for Histological WSI Normalization."

    """
    alpha = tf.Variable(init_varphi[0], dtype='float32')
    beta = tf.Variable(init_varphi[1], dtype='float32')
    w = [tf.Variable(1.0, dtype='float32'), tf.Variable(1.0, dtype='float32'), tf.constant(1.0)]

    sca_mat = tf.stack((tf.cos(alpha) * tf.sin(beta), tf.cos(alpha) * tf.cos(beta), tf.sin(alpha)), axis=1)
    cd_mat = tf.matrix_inverse(sca_mat)

    s = tf.matmul(input_od, cd_mat) * w
    h, e, b = tf.split(s, (1, 1, 1), axis=1)

    l_p1 = tf.reduce_mean(tf.square(b))
    l_p2 = tf.reduce_mean(2 * h * e / (tf.square(h) + tf.square(e)))
    l_b = tf.square((1 - eta) * tf.reduce_mean(h) - eta * tf.reduce_mean(e))
    l_e = tf.square(gamma - tf.reduce_mean(s))

    objective = l_p1 + lambda_p * l_p2 + lambda_b * l_b + lambda_e * l_e
    target = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(objective)

    return target, cd_mat, w
