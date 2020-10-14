# -*- coding: utf-8 -*-
"""
@author: Michael
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visual_Odometry:
    def __init__(self,frames,method,featureNumber):
        self.method = method
        self.featureNumber = featureNumber
        self.frames = frames

    #capture two sequential image starting from time_n
    def readimage():
        imglist = []
        for index,entries in enumerate(os.listdir(path)):
            if index <time:
                pass
            elif index >time+1:
                break
            else:
                img = cv2.imread(os.path.join(path,entries))
                img = np.asarray(img[:,:])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imglist.append(img)   
        return imglist

    #Extracting features and descriptors from the image(t) and image(t+1)
    def featureExtraction():
        

        def Orb_feature_detection(image_pair):
            orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
            kp0 = orb.detect(image_pair[0],None)
            kp1 = orb.detect(image_pair[1],None)
            kp0,des0 = orb.compute(image_pair[0],kp0)
            kp1,des1 = orb.compute(image_pair[1],kp1)
            return kp0 ,kp1 ,des0, des1

        def Surf_feature_detection(image_pair):
            surf = cv2.SURF(500)
            kp0 = surf.detect(image_pair[0],None)
            kp1 = surf.detect(image_pair[1],None)
            kp0,des0 = surf.compute(image_pair[0],kp0)
            kp1,des1 = surf.compute(image_pair[1],kp1)
            return kp0 ,kp1 ,des0, des1

        def Akaze_feature_detection(image_pair):
            kaze = cv2.AKAZE_create(500)
            kp0 = kaze.detect(image_pair[0],None)
            kp1 = kaze.detect(image_pair[1],None)
            kp0,des0 = kaze.compute(image_pair[0],kp0)
            kp1,des1 = kaze.compute(image_pair[1],kp1)
            return kp0 ,kp1 ,des0, des1    

        def Fast_feature_detection(image_pair):
            fast = cv2.FastFeatureDetector(500)
            kp0 = fast.detect(image_pair[0],None)
            kp1 = fast.detect(image_pair[1],None)
            kp0,des0 = fast.compute(image_pair[0],kp0)
            kp1,des1 = fast.compute(image_pair[1],kp1)
            return kp0 ,kp1 ,des0, des1    
        
        if self.method == 'Orb':
            

images = readimage('./rgb',0)   
kep,kep0,des,des0 = Surf_feature_detection(images)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(np.float32(des),np.float32(des0),k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        pts1.append(kep[m.trainIdx].pt)
        pts2.append(kep0[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

E, mask = cv2.findEssentialMat(pts1 ,pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.1)

print(E)

points, R, t, mask = cv2.recoverPose(E, pts1, pts2)

print(R)
print(T)