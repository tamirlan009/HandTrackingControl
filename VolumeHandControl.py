import cv2 as cv
import time
import numpy as np
import math

from HandTrackingModule import HandDetector

###########################

wCam, hCam = 640, 480

##########################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

cTime = 0
pTime = 0

detector = HandDetector(detectionCon=0.8, trackConfidence=0.8)

while True:

    success, img = cap.read()

    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv.circle(img, (x1, y1), 10, (220, 20, 60), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (220, 20, 60), cv.FILLED)
        cv.circle(img, (cx, cy), 13, (240, 128, 128), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 192, 203), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)
        if length < 50:
            cv.circle(img, (cx, cy), 13, (50, 205, 50), cv.FILLED)

    cTime = time.time()

    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_COMPLEX,
               1, (255, 0, 0), 2)

    cv.imshow('result', img)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
