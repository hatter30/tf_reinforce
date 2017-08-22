import tensorflow as tf


def standardize(tensor):
    mean, std_dev = tf.nn.moments(tensor, [0], shift=None, name="reward_moments")
    return (tensor - mean) / std_dev