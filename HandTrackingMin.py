import cv2 as cv
import mediapipe as mp

import time

mpHand = mp.solutions.hands
hands = mpHand.Hands(min_tracking_confidence=0.8,min_detection_confidence=0.8)

mpDraw = mp.solutions.drawing_utils
mpDraw.DrawingSpec.circle_radius = 10
cap = cv.VideoCapture(0)

pTime = 0
cTime = 0


while True:

    succes, img = cap.read()

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handsLM in result.multi_hand_landmarks:
            for id, lm in enumerate(handsLM.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                # if id == 0:
                #     cv.circle(img,(cx,cy), 10, (0,0,0))
                mpDraw.draw_landmarks(img, handsLM, mpHand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

    cv.imshow('result', img)

    if cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()
