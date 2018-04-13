
import cv2  
import numpy as np  
from matplotlib import pyplot as plt  
  
  
def sameSize(img1, img2):
    rows, cols, dpt = img2.shape  
    dst = img1[:rows,:cols]  
    return dst  
  
def laplace_fuse(apple, orange):
    G = apple.copy()  
    gp_apple = [G]  
    for i in xrange(6):  
        G = cv2.pyrDown(G)  
        gp_apple.append(G)  
    
    G = orange.copy()  
    gp_orange = [G]  
    for j in xrange(6):  
        G = cv2.pyrDown(G)  
        gp_orange.append(G)  
    
    lp_apple = [gp_apple[5]]  
    for i in xrange(5,0,-1):  
        GE = cv2.pyrUp(gp_apple[i])  
        L = cv2.subtract(gp_apple[i-1], sameSize(GE,gp_apple[i-1]))  
        lp_apple.append(L)  
    
    lp_orange = [gp_orange[5]]  
    for i in xrange(5,0,-1):  
        GE = cv2.pyrUp(gp_orange[i])  
        L = cv2.subtract(gp_orange[i-1], sameSize(GE,gp_orange[i-1]))  
        lp_orange.append(L)  
    
    LS = []  
    for la,lb in zip(lp_apple,lp_orange):  
        rows,cols,dpt = la.shape
        ls = (la + lb)/2
        LS.append(ls)  
    
    ls_reconstruct = LS[0]  
    for i in xrange(1,6):  
        ls_reconstruct = cv2.pyrUp(ls_reconstruct)  
        ls_reconstruct = cv2.add(sameSize(ls_reconstruct,LS[i]), LS[i])  
    
    return ls_reconstruct
