import numpy as np
import cv2 as cv

def get_landmark_counts(img, detector='SIFT'):
    feat_detector = cv.SIFT_create() if detector == 'SIFT' else cv.ORB_create()
    kp, des = feat_detector.detectAndCompute(img, None)
    if des is None:
        return 0
    else:
        return len(des)