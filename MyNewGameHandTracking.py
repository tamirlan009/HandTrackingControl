import cv2 as cv
import time

from HandTrackingModule import HandDetector


pTime = 0
cTime = 0

cap = cv.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, trackConfidence=0.8)

while True:
    success, img = cap.read()

    img = detector.findHands(img)

    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        print(lmList[10])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

    cv.imshow('result', img)

    if cv.waitKey(1) == ord('q'):
        break