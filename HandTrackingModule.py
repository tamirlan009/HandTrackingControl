import cv2 as cv
import mediapipe as mp

import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackConfidence = trackConfidence

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.detectionCon, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img, draw = True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handsLM in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handsLM,
                                               self.mpHand.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handNo = 0, draw = True):

        lmList = []

        if self.result.multi_hand_landmarks:
            myHand =  self.result.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img,(cx,cy), 10, (255,0,255), cv.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, trackConfidence=0.8)
    while True:
        success, img = cap.read()

        img = detector.findHands(img)

        lmList = detector.findPosition(img)

        if len(lmList)!=0:
            print(lmList[10])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

        cv.imshow('result', img)

        if cv.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()



















