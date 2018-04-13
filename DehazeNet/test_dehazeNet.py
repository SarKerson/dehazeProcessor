from scipy.io import loadmat
from dehazeNet import dehazeNet
import cv2
import tensorflow as tf
import numpy as np
import time

MAT_PATH = '/home/sar/SarKerson/dehaze/DnCNN/DeCNN/data/model_15_12b_64/model_15_12b_64-epoch-300.mat'
TR_PATH = 'pre_t.png'

# def save_info(layer, str):
#     t1 = layer[0, :, :, 0]
#     cv2.imwrite(str + '-1.png', t1)
#     t1 = layer[0, :, :, 9]
#     cv2.imwrite(str + '-10.png', t1)
#     t1 = layer[0, :, :, 29]
#     cv2.imwrite(str + '-30.png', t1)
#     t1 = layer[0, :, :, 63]
#     cv2.imwrite(str + '-64.png', t1)
#     print(t1[:,11])

# build the graph
graph = tf.Graph()
with graph.as_default():
    data = loadmat(MAT_PATH)
    data = data['net']

    # read meta info
    meta = data['meta']
    learning_rate = meta[0][0][0][0][1]
    inputSize = meta[0][0][0][0][2]
    image_size = np.squeeze(inputSize)
    input_maps = tf.placeholder(tf.float32, [None, 300, 400, 1])
    net = dehazeNet(data, input_maps)
    output = {}
    output['1'] = net['layer1']
    output['5'] = net['layer5']
    output['11'] = net['layer11']
    output['12'] = net['layer12']
    output['14'] = net['layer14']
    output['last'] = net['layer24'][0]

img = cv2.imread(TR_PATH, 0)
img = img / 255.0
img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_LINEAR)
# print(img.shape)
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 3)
#
# run the graph
with tf.Session(graph=graph) as sess:
    time1 = time.time()
    output = sess.run(output, feed_dict={input_maps: img})

    output_trm = np.reshape(output['last'], [300, 400])
    output_trm = output_trm * 255.0
# cv2.imwrite('out.png', output_trm)