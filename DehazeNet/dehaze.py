import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from dehazeNet import dehazeNet
import cv2
import time
from ImageUtils.imgworker import cal_dark_channel_fast, dehazeFun, guidedfilter, em_A_color, auto_tune, boundcon
MAT_PATH = './model/model_15_12b_64-epoch-300.mat'
IMG_PATH = './input/d.jpg'
# build the graph
graph = tf.Graph()
with graph.as_default():
    data = loadmat(MAT_PATH)
    data = data['net']

    meta = data['meta']
    learning_rate = meta[0][0][0][0][1]
    inputSize = meta[0][0][0][0][2]
    image_size = np.squeeze(inputSize)
    input_maps = tf.placeholder(tf.float32, [None, 300, 400, 1])
    net, _ = dehazeNet(data, input_maps)
    output = net['layer24'][0]

sess = tf.Session(graph = graph)

img = cv2.imread(IMG_PATH)
img = img / 255.0
img = img.astype('float32')
img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_LINEAR)

def refine_transmission_map_net(sess, t_map):
    global output
    output = sess.run(output, feed_dict={input_maps: t_map})
    return np.reshape(output, [300, 400])

def hazee_free_net(sess, source_img):
    d_map = cal_dark_channel_fast(source_img, patch_size=15)
    A = em_A_color(source_img, d_map)
    t_map = boundcon(hazeImg=source_img * 255.0, A=A * 255.0, C0=30.0, C1=300.0, patch_size=3)  # 255
    t_map = np.expand_dims(t_map, 0)
    t_map = np.expand_dims(t_map, 3)
    t_map = refine_transmission_map_net(sess=sess, t_map=t_map)
    t_map = np.reshape(t_map, [300, 400])
    t_map = guidedfilter(source_img, t_map, 10, 0.001)
    cv2.imwrite("a__.png", t_map * 255.0)
    return dehazeFun(source_img * 255.0, t_map, A * 255.0, 0.75)

time1 = time.time()
haze_free_ = hazee_free_net(sess=sess, source_img=img)
time2 = time.time()
print(time2 - time1)
cv2.imwrite("out!!!!.png", auto_tune(haze_free_) * 255.0)
cv2.imwrite("in!!!.png", img * 255.0)
