import tensorflow as tf
from os import path
import cv2
import numpy as np


from ImageUtils.imgworker import cal_dark_channel_fast, dehazeFun, guidedfilter, em_A_color, auto_tune, boundcon
IMG_PATH = './input/d.jpg'

graph = tf.Graph()
with graph.as_default():
    pb = tf.GraphDef()
    with open('./model/new-model.pb', "rb") as fin:
        pb.ParseFromString(fin.read())  # binary
    imports = tf.import_graph_def(pb, name="")

    input = graph.get_tensor_by_name("input:0")
    outputs = graph.get_tensor_by_name("conv12/output:0")

img = cv2.imread(IMG_PATH)
img = img / 255.0
img = img.astype('float32')
img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_LINEAR)

d_map = cal_dark_channel_fast(img, patch_size=15)
A = em_A_color(img, d_map)
t_map = boundcon(hazeImg=img * 255.0, A=A * 255.0, C0=30.0, C1=300.0, patch_size=3)  # 255
cv2.imwrite("t_pre.png", t_map * 255.0)
t_map = np.expand_dims(t_map, 0)
t_map = np.expand_dims(t_map, 3)

with tf.Session(graph=graph) as sess:
    out = sess.run(outputs, feed_dict={input: t_map})
    print(out)
