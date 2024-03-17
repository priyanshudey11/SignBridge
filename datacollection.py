import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
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
