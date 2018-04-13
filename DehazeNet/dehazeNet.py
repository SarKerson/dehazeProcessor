import tensorflow as tf
import numpy as np

def dehazeNet(data, input_maps):

    # read layer info
    layers = data['layers']
    layers = layers[0][0][0][0:-1]    # from the first to the last second layer
    current = input_maps
    network = {}
    data_dict = {}
    layer_num = 1
    for layer in layers:
        name = layer['name'][0][0][0]
        layer_type = layer['type'][0][0][0]
        if layer_type == 'conv':
            data_dict['conv' + str(layer_num)] = {}
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer['stride'][0][0][0]
            kernel, bias = layer['weights'][0][0][0]
            if len(kernel.shape) == 3:
                kernel = tf.expand_dims(tf.constant(kernel), 3)
            else:
                kernel = tf.constant(kernel)
            bias = np.squeeze(bias).reshape(-1)
            data_dict['conv' + str(layer_num)]['weights'] = kernel
            data_dict['conv' + str(layer_num)]['biases'] = bias
            data_dict['conv' + str(layer_num)]['stride'] = np.array([1, stride[0], stride[0], 1])
            conv = tf.nn.conv2d(current, kernel,
                                strides=(1, stride[0], stride[0], 1), padding=padding)
            current = tf.nn.bias_add(conv, bias)
            # print name, 'stride:', stride, 'kernel size:', np.shape(kernel)
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
            layer_num += 1
            # print name + " " + layer_type
        elif layer_type == 'pool':
            stride = layer['stride'][0][0][0]
            pool = layer['pool'][0][0][0]
            current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                     strides=(1, stride[0], stride[0], 1), padding='SAME')
            # print name, 'stride:', stride
        elif layer_type == 'bnorm':
            epsilon = layer['epsilon'][0][0][0]
            scale, offset, _ = layer['weights'][0][0][0]
            scale = np.transpose(scale)[0]
            offset = np.transpose(offset)[0]
            axis = [0, 1, 2]
            mean, var = tf.nn.moments(current, axes=axis)
            print(mean.get_shape)
            print(var.get_shape)
            data_dict['conv' + str(layer_num)]['scale'] = scale
            data_dict['conv' + str(layer_num)]['offset'] = offset
            data_dict['conv' + str(layer_num)]['epsilon'] = epsilon
            scale = tf.constant(scale)
            offset = tf.constant(offset)
            current = tf.nn.batch_normalization(current, mean=mean, variance=var,
                                    offset=offset, scale=scale, variance_epsilon=epsilon)

            # print name + " " + layer_type

        network[name] = current

    return network, data_dict

def create_dehazeNet(X):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', shape = [3, 3, 1, 64])
        conv = tf.nn.conv2d(X,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias)

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape = [3, 3, 64, 64])
        conv = tf.nn.conv2d(conv1,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)

    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights', shape = [3, 3, 64, 64])
        conv = tf.nn.conv2d(conv2,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias)

    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv3,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias)

    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv4,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias)

    with tf.variable_scope('conv6') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv5,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        axis = [0, 1, 2]
        mean = tf.reduce_mean(bias, axis=axis)
        var_ = tf.subtract(bias, mean)
        var_ = tf.multiply(var_, var_)
        var = tf.reduce_mean(var_, axis=axis)
        scale = tf.get_variable('scale',
                                shape=[64])
        offset = tf.get_variable('offset',
                                 shape=[64])
        bn = tf.nn.batch_normalization(bias, mean=mean, variance=var,
                                            offset=offset, scale=scale, variance_epsilon=1.0e-05)
        conv6 = tf.nn.relu(bn)

    with tf.variable_scope('conv7') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv6,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv7 = tf.nn.relu(bias)

    with tf.variable_scope('conv8') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv7,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv8 = tf.nn.relu(bias)

    with tf.variable_scope('conv9') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv8,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv9 = tf.nn.relu(bias)

    with tf.variable_scope('conv10') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv9,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv10 = tf.nn.relu(bias)

    with tf.variable_scope('conv11') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 64])
        conv = tf.nn.conv2d(conv10,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64])
        bias = tf.nn.bias_add(conv, biases)
        conv11 = tf.nn.relu(bias)

    with tf.variable_scope('conv12') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 64, 1])
        conv = tf.nn.conv2d(conv11,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[1])
        conv12 = tf.nn.bias_add(conv, biases, name='output')

    return conv12




