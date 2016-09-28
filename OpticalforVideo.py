import cv2
import numpy as np
import os
import sys
from PIL import Image
from opticalflow import OpticalFlow
def Video2flow(videoname):
    Flow = []
    cap = cv2.VideoCapture(videoname)
    if(cap.isOpened()==False):
        raise Error
    ret,frame1 = cap.read()
    prvs = cv2.cvtColor(np.array(frame1),cv2.COLOR_GRB2GRAY)
    while(ret):
        ret,frame2 = cap.read()
        next = cv2.cvtColor(np.array(frame2),cv2.COLOR_GRB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next,  0.5, 3, 15, 3, 5, 1.2, 0)
        Flow.append(flow)
        if(ret==0):
            break
        prvs=next
    return Flow    
        
if __name__ == '__main__':
    Flow = Video2flow(sys.argv[1])