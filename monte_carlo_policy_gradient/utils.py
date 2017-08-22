import tensorflow as tf


def standardize(tensor):
    mean, std_dev = tf.nn.moments(self.tf_discounted_epr, [0], shift=None, name="reward_moments")
    return (tensor - mean) / std_dev