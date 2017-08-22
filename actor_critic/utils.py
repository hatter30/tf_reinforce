import tensorflow as tf


def standardize(tensor):
    mean, std_dev = tf.nn.moments(tensor, [0], shift=None, name="reward_moments")
    return (tensor - mean) / std_dev
    
def discount_rewards(tensor):
    discount_f = lambda a, v: a*0.99 + v
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tensor,[True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
    return tf_discounted_r