import tensorflow as tf


def get_slice(d, fold, reverse = False):

    if reverse:
        return [example for example in d if example['fold'] != fold]
    else:
        return [example for example in d if example['fold'] == fold]

def to_dict_of_list(data_dict):
    return {key: [i[key] for i in data_dict] for key in data_dict[0]}

def replace_nan_w_zero(value):
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(value)), dtype=value.dtype)
    return tf.math.multiply_no_nan(value, value_not_nan)

def get_scaler(maxs, mins):
    def scale(x):
        return (x - mins) / (maxs - mins)

    return scale

def get_dict_mod(key, fn):

    def dict_mod_fn(example):
        example[key] = fn(example[key])
        return example

    return dict_mod_fn
