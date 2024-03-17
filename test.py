import csv
import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import datacollection
class DepthwiseConv2DIgnoreGroups(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Ignore groups argument
        super().__init__(*args, **kwargs)

# Load the model with the custom object
model = load_model(datacollection.py, custom_objects={'DepthwiseConv2D': DepthwiseConv2DIgnoreGroups})
class HandGestureClassifier:
    def __init__(self, model_path, labels, video_source=0):
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier(model_path, labels)
        self.cap = cv2.VideoCapture(video_source)
        self.labels = labels
        self.offset = 20
        self.imgSize = 300

    def classify_gesture(self):
        while True:
            success, img = self.cap.read()
            imgOutput = img.copy()
            hands, img = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8)*255
                imgCrop = img[y-self.offset:y + h + self.offset, x-self.offset:x + w + self.offset]
                imgWhite, prediction, index = self.process_image(imgCrop, imgWhite, w, h)
                self.display_output(imgOutput, imgCrop, imgWhite, x, y, w, h, prediction, index)

    def process_image(self, imgCrop, imgWhite, w, h):
        aspectRatio = h / w
        if aspectRatio > 2:
            imgWhite, prediction, index = self.process_tall_image(imgCrop, imgWhite, w, h)
        else:
            imgWhite, prediction, index = self.process_wide_image(imgCrop, imgWhite, w, h)
        return imgWhite, prediction, index

    def process_tall_image(self, imgCrop, imgWhite, w, h):
        k = self.imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
        wGap = math.ceil((self.imgSize-wCal)/2)
        imgWhite[:, wGap: wCal + wGap] = imgResize
        prediction , index = self.classifier.getPrediction(imgWhite, draw= False)
        return imgWhite, prediction, index

    def process_wide_image(self, imgCrop, imgWhite, w, h):
        k = self.imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
        hGap = math.ceil((self.imgSize - hCal) / 2)
        imgWhite[hGap: hCal + hGap, :] = imgResize
        prediction , index = self.classifier.getPrediction(imgWhite, draw= False)
        return imgWhite, prediction, index

    def display_output(self, imgOutput, imgCrop, imgWhite, x, y, w, h, prediction, index):
        cv2.rectangle(imgOutput,(x-self.offset,y-self.offset-70),(x -self.offset+400, y - self.offset+60-50),(0,255,0),cv2.FILLED)  
        cv2.putText(imgOutput,self.labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
        cv2.rectangle(imgOutput,(x-self.offset,y-self.offset),(x + w + self.offset, y+h + self.offset),(0,255,0),4)   
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)
        cv2.imshow('Image', imgOutput)
        cv2.waitKey(1)

if __name__ == "__main__":
    model_path = "Model/keras_model.h5"
    labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]
    classifier = HandGestureClassifier(model_path, labels)
    classifier.classify_gesture()

'''import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Data/Okay"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        aspectRatio = h / w
        k = imgSize / (h if aspectRatio > 1 else w)

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgResize = cv2.resize(imgCrop, (math.ceil(k * w), imgSize) if aspectRatio > 1 else (imgSize, math.ceil(k * h)))
        imgResizeShape = imgResize.shape

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if aspectRatio > 1:
            wCal = imgResizeShape[1]
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap: wGap + wCal] = imgResize
        else:
            hCal = imgResizeShape[0]
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap: hGap + hCal, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
'''
