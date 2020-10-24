# -*- coding: utf-8 -*-
"""
@author: Michael
"""

import os
import cv2
import numpy as np


class Visual_Odometry:
    def __init__(self, framesPath, method, featureNumber):
        self.method = method
        self.featureNumber = featureNumber
        self.framesPath = framesPath
        self.time = 0

    def increment_time(self):
        self.time += 1

    # capture two sequential image starting from time_n

    def readimage(self):
        imglist = []
        for index, entries in enumerate(os.listdir(self.framesPath)):
            if index < self.time:
                pass
            elif index > self.time + 1:
                break
            else:
                img = cv2.imread(os.path.join(self.framesPath, entries))
                img = np.asarray(img[:, :])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imglist.append(img)
        return imglist

    # Extracting features and descriptors from the image(t) and image(t+1)

    def featureExtraction(self):
        def Orb_feature_detection():
            orb = cv2.ORB_create(
                edgeThreshold=15,
                patchSize=31,
                nlevels=8,
                fastThreshold=20,
                scaleFactor=1.2,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                firstLevel=0,
                nfeatures=self.featureNumber,
            )
            return orb

        def Sift_feature_detection():
            surf = cv2.SIFT(self.featureNumber)
            return surf

        def Akaze_feature_detection():
            kaze = cv2.AKAZE_create(self.featureNumber)
            return kaze

        def Fast_feature_detection():
            fast = cv2.FastFeatureDetector(self.featureNumber)
            return fast

        def keypointsANDdescriptors(featureExtraction_method):
            successivepoints = []
            for image in self.readimage():
                keypoint = featureExtraction_method.detect(image, None)
                keypoint, descriptor = featureExtraction_method.compute(image, keypoint)
                successivepoints.append(keypoint)
                successivepoints.append(descriptor)
            return successivepoints

        if self.method == "Orb":
            kep = keypointsANDdescriptors(Orb_feature_detection())
        elif self.method == "Sift":
            kep = keypointsANDdescriptors(Sift_feature_detection())
        elif self.method == "Kaze":
            kep = keypointsANDdescriptors(Akaze_feature_detection())
        elif self.method == "Fast":
            kep = keypointsANDdescriptors(Fast_feature_detection())

        return kep

    def keypointsMatching(self):
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(
            np.float32(self.featureExtraction()[3]),
            np.float32(self.featureExtraction()[1]),
            2,
        )

        pts0 = []
        pts1 = []

        for i, (m, n) in enumerate(knn_matches):
            if m.distance < 0.7 * n.distance:

                pts0.append(self.featureExtraction()[0][m.trainIdx].pt)
                pts1.append(self.featureExtraction()[2][n.queryIdx].pt)

        return np.int32(pts0), np.int32(pts1)

    def EssentialMatrix(self):
        pts1, pts2 = self.keypointsMatching()
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.001,
        )
        return E, mask

    def RotationalAndTranslational(self):
        for __ in os.listdir(self.framesPath):
            self.featureExtraction()
            pts1, pts2 = self.keypointsMatching()
            E, mask = self.EssentialMatrix()
            __, R, t, __ = cv2.recoverPose(E, pts1, pts2)
            yield R, t
            self.increment_time


# vo = Visual_Odometry("./data_1", "Orb", 500)
# for Rotational, Translational in vo.RotationalAndTranslational():
#     print(Rotational)
