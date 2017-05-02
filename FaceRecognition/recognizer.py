import cv2
from PIL import Image ,ImageFont, ImageDraw
import numpy as np


def select(size, x, y, t, color, name, picture, thickness):
    cv2.line(picture, (x, y), (x + size, y), color, thickness)
    cv2.line(picture, (x, y), (x, y + size), color, thickness)

    cv2.line(picture, (x + t, y), (x + t - size, y), color, thickness)
    cv2.line(picture, (x + t, y), (x + t, y + size), color, thickness)

    cv2.line(picture, (x, y + t), (x + size, y + t), color, thickness)
    cv2.line(picture, (x, y + t), (x, y + t - size), color, thickness)

    cv2.line(picture, (x + t, y + t), (x + t - size, y + t), color, thickness)
    cv2.line(picture, (x + t, y + t), (x + t, y + t - size), color, thickness)
    # cv2.rectangle(picture, (x, y), (x + t, y + t), color, thickness)
    cv2.putText(picture, name, (x, y-5), font, 2, color, thickness, cv2.LINE_AA)


def clearCapture(capture):
    capture.release()
    cv2.destroyAllWindows()

def countCameras():
    videos = []
    n = 0
    open = True

    while open:
        try:
            cap = cv2.VideoCapture(n)
            ret, frame = cap.read()
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clearCapture(cap)
            n += 1
        except:
            clearCapture(cap)
            open = False
            break
    return n

def getCameras():
    videos=[]
    cameras = countCameras()
    print(cameras)
    for i in range(cameras):
        videos.append(cv2.VideoCapture(i))
    return videos



recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
scale_factor = 1.2
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    faces = faceCascade.detectMultiScale(gray, scale_factor, 5)
    for(x, y, w, h) in faces:
        t = w
        if h > w:
            t = h

        FoundId, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if(confidence < 60):
            if(FoundId == 0):
                FoundId = "Judah"
        else:
            FoundId = "?"
            # Id = input('enter your id')
            # FoundId = Id

        # Id = "?"
        select(10, x, y, t, (255, 255, 255), str(FoundId), image, 1)
    cv2.imshow('VIDEO',image)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()