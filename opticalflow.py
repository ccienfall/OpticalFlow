import cv2
import numpy as np
import os
from PIL import Image
import sys

def OpticalFlow(prvs,next):
    prvs = np.asarray(prvs,dtype = np.uint8)
    prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
    next = np.asarray(next,dtype=np.uint8)
    next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5,3,15,3,5,1.2,0)
    return flow


if __name__ == '__main__':
    prvs = Image.open(sys.argv[1],'r')
    next = Image.open(sys.argv[2],'r')
    flow = OpticalFlow(prvs,next)
    print(flow.shape)
    flow = flow.reshape(-1,2)
    #print(flow)
    #f = open('flow.txt','w')
    #for i in flow:
    #    f.writelines(str(i))

