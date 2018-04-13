#from scipy.misc import imsave, imshow, imread

import numpy as np
import math
import cv2
import tensorflow as tf
from guided_filter.core.filters import FastGuidedFilter

import time

imgworker_out_img=None
imgworker_input_img=None

def guidedfilter(I, p, r, eps):
    gf = FastGuidedFilter(I,r,eps)
    ret = gf.filter(p)
    return ret

def cal_dark_channel_single(source_img, patch_size=15):
    UL = np.array(source_img)
    UR = np.array(source_img)
    LL = np.array(source_img)
    LR = np.array(source_img)

    size_x = source_img.shape[0]
    size_y = source_img.shape[1]

    dk = np.zeros([size_x, size_y])
    # calculate the LUT

    r_x = int(math.ceil(size_x/float(patch_size)))
    r_y = int(math.ceil(size_y/float(patch_size)))

    for x in range(r_x):
        for y in range(r_y):
            s_x = (x+1)*patch_size-1
            e_x = x*patch_size-1
            s_y = (y+1)*patch_size-1
            e_y = y*patch_size-1
            for _x in range(s_x, e_x, -1):
                for _y in range(s_y, e_y, -1):
                    if _x>=size_x or _y>=size_y:
                        continue
                    UL[_x, _y] = min(UL[min(_x + 1, s_x, size_x-1), _y],UL[_x, min(_y + 1, s_y, size_y-1)])

    for x in range(r_x):
        for y in range(r_y):
            s_x = (x+1)*patch_size-1
            e_x = x*patch_size-1
            s_y = y*patch_size
            e_y = (y+1)*patch_size
            for _x in range(s_x, e_x, -1):
                for _y in range(s_y, e_y):
                    if _x>=size_x or _y>=size_y:
                        continue
                    UR[_x, _y] = min(UR[min(_x+1, s_x, size_x-1), _y], UR[_x, max(_y-1, s_y)])

    for x in range(r_x):
        for y in range(r_y):
            s_x = x*patch_size
            e_x = (x+1)*patch_size
            s_y = (y+1)*patch_size-1
            e_y = y*patch_size-1
            for _x in range(s_x, e_x):
                for _y in range(s_y, e_y, -1):
                    if _x>=size_x or _y>=size_y:
                        continue
                    LL[_x, _y] = min(LL[max(_x-1, s_x), _y], LL[_x, min(_y+1, s_y, size_y-1)])
    
    for x in range(r_x):
        for y in range(r_y):
            s_x = x*patch_size
            e_x = (x+1)*patch_size
            s_y = y*patch_size
            e_y = (y+1)*patch_size
            for _x in range(s_x, e_x):
                for _y in range(s_y, e_y):
                    if _x>=size_x or _y>=size_y:
                        continue
                    LR[_x, _y] = min(LR[max(_x - 1, s_x), _y], LR[_x, max(_y - 1, s_y)])
    
    for x in range(size_x):
        for y in range(size_y):
            s_x = min(max(x, patch_size/2), size_x-1-patch_size/2)
            s_y = min(max(y, patch_size/2), size_y-1-patch_size/2)
            dk[x,y] = min(UL[s_x-patch_size/2, s_y-patch_size/2], UR[s_x-patch_size/2, s_y+patch_size/2], LL[s_x+patch_size/2, s_y-patch_size/2], LR[s_x+patch_size/2, s_y+patch_size/2])
    return dk


def cal_dark_channel_fast(source_img, patch_size=15):
    full = np.min(source_img, axis=2)
    return cal_dark_channel_single(full, patch_size)


def cal_dark_channel(sess, source_img, patch_size=15):
    # tensorflow only has max_pooling so we could only inverse
    global imgworker_out_img
    global imgworker_input_img
    if imgworker_out_img==None:
        imgworker_input_img = tf.placeholder(tf.float32, (source_img.shape[0],source_img.shape[1],3))
        temp_img = tf.reduce_min(imgworker_input_img, axis=2)
        temp_img = tf.reshape(temp_img, [1, source_img.shape[0],source_img.shape[1], 1])
        temp_img = 1-temp_img
        imgworker_out_img = tf.nn.max_pool(temp_img, ksize=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], padding='SAME')
        # inverse back
        imgworker_out_img = 1-imgworker_out_img
        imgworker_out_img = tf.reshape(imgworker_out_img, [source_img.shape[0],source_img.shape[1]])

    return sess.run(imgworker_out_img, feed_dict={imgworker_input_img:source_img})

def em_A_color(source_img, d_map, percent = 0.001):
    temp_t = np.reshape(np.array(d_map),[-1])
    take_count = int(len(temp_t)*0.001)
    temp_t.sort()
    d_instbasic = temp_t[len(temp_t)-take_count-1]
    bix = 0
    biy = 0
    for x in range(source_img.shape[0]):
        for y in range(source_img.shape[1]):
            if d_map[x,y]>d_instbasic:
                if (source_img[x,y,0]+source_img[x,y,1]+source_img[x,y,2])>(source_img[bix,biy,0]+source_img[bix,biy,1]+source_img[bix,biy,2]):
                    bix = x
                    biy = y

    return source_img[bix,biy]

def em_haze_free(source_img, A, t_map):
    ret = source_img-A
    t_shape = [t_map.shape[0],t_map.shape[1],1]
    t_map = np.reshape(t_map, t_shape)
    t_map = np.concatenate((t_map, np.ones(t_shape)*0.1), axis=2 )
    t_map = np.max(t_map, axis=2)
    ret[:,:,0]/=t_map
    ret[:,:,1]/=t_map
    ret[:,:,2]/=t_map
    return ret+A

def dehazeFun(sourceImg, t, A, delta):
    t = np.maximum(np.abs(t), 0.0001) * delta
    R = np.divide(sourceImg[:, :, 0] - A[0], t)
    R = np.add(R, A[0])
    R = np.expand_dims(R, axis=2)
    G = np.divide(sourceImg[:, :, 1] - A[1], t)
    G = np.add(G, A[1])
    G = np.expand_dims(G, axis=2)
    B = np.divide(sourceImg[:, :, 2] - A[2], t)
    B = np.add(B, A[2])
    B = np.expand_dims(B, axis=2)
    dstImg = np.concatenate([R, G, B], axis=2)
    return dstImg


def em_transmission_map(sess, source_img, patch_size=15, w=0.95):
    return 1-w*cal_dark_channel(sess, source_img, patch_size)

def em_transmission_map_d(d_map, w=1):
    #w=0.95
    return 1-w*d_map

def auto_tune_single(I, percent):
    I_c = np.sort(np.reshape(np.array(I),[-1]))
    I_min = I_c[int(percent*I.shape[0]*I.shape[1])]
    I_max = I_c[int((1-percent)*I.shape[0]*I.shape[1])]
    I = np.clip(I, I_min, I_max)
    I_dif = I_max-I_min
    return (I-I_min)/I_dif 

def equlize_hist(source_img):
    source_img=np.clip(source_img, 0, 1)
    source_img=source_img*255.0
    source_img=source_img.astype('uint8')
    ret_0 = np.reshape(cv2.equalizeHist(source_img[:,:,0]), [source_img.shape[0], source_img.shape[1], 1])
    ret_1 = np.reshape(cv2.equalizeHist(source_img[:,:,1]), [source_img.shape[0], source_img.shape[1], 1])
    ret_2 = np.reshape(cv2.equalizeHist(source_img[:,:,2]), [source_img.shape[0], source_img.shape[1], 1])
    ret = np.concatenate((ret_0,ret_1,ret_2), axis=2)
    ret = ret.astype('float32')
    ret = ret/255.0
    return ret

def auto_tune(I, percent=0.001):
    o_0 = np.reshape(auto_tune_single(I[:,:,0],percent), [I.shape[0],I.shape[1],1])
    o_1 = np.reshape(auto_tune_single(I[:,:,1],percent), [I.shape[0],I.shape[1],1])
    o_2 = np.reshape(auto_tune_single(I[:,:,2],percent), [I.shape[0],I.shape[1],1])
    # print(o_0[0,0,0])
    return np.concatenate((o_0,o_1,o_2),axis=2)

def boundcon(hazeImg, A, C0, C1, patch_size=3):  # hazeImg, A -> 0 ~ 255.0,
    if len(A) == 1:
        A = A * np.ones([3, 1])
    C0 = C0 * np.ones([3, 1])
    C1 = C1 * np.ones([3, 1])
    t_r = np.maximum((A[0] - hazeImg[:,:,0]) / (A[0] - C0[0]),
                     (hazeImg[:,:,0] - A[0]) / C1[0] - A[0]);
    t_r = np.expand_dims(t_r, axis=2)
    t_g = np.maximum((A[1] - hazeImg[:,:,1]) / (A[1] - C0[1]),
                     (hazeImg[:,:,1] - A[1]) / C1[1] - A[1]);
    t_g = np.expand_dims(t_g, axis=2)
    t_b = np.maximum((A[2] - hazeImg[:,:,2]) / (A[2] - C0[2]),
                     (hazeImg[:,:,2] - A[2]) / C1[2] - A[2]);
    t_b = np.expand_dims(t_b, axis=2)
    t_b = np.max(np.concatenate([t_r, t_g, t_b], axis=2), axis=2)
    return t_b

def haze_free(sess, source_img, patch_size=15):
    #d_map = cal_dark_channel_fast(source_img, patch_size=15)
    d_map = cal_dark_channel(sess, source_img, patch_size)
    t_map = em_transmission_map_d(d_map)
    t_map = guidedfilter(source_img, t_map, 20, 0.001)
    A = em_A_color(source_img, d_map)
    return em_haze_free(source_img, A, t_map)

