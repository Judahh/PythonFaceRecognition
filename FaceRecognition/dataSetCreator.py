import cv2, os
import numpy as np
from PIL import Image

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

def getLastSampleNumber(path, receivedId):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    last=0
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #getting the Id from the image
        currentId = int(os.path.split(imagePath)[-1].split(".")[1])
        if(currentId == receivedId):
            current = int(os.path.split(imagePath)[-1].split(".")[2])
            if(last < current):
                last = current
    return last

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=faceCascade.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

recognizer = cv2.face.createLBPHFaceRecognizer()
try:
    recognizer.load('trainner/trainner.yml')
except:
    recognizer.save('trainner/trainner.yml')
cascadePath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
scale_factor = 1.2
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

Id = input('enter your id')

sampleNum = getLastSampleNumber('dataSet', int(Id))
print(sampleNum)
first = sampleNum
while (True):
    ret, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scale_factor, 5)
    for (x, y, w, h) in faces:
        t = w
        if h > w:
            t = h

        try:
            FoundId, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if (confidence < 50):
                if (FoundId == 0):
                    FoundId = "Judah"
            else:
                FoundId = "?"
                cv2.imwrite("dataSet/User." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                print("New " + "dataSet/User." + Id + '.' + str(sampleNum) + ".jpg")
                faces, Ids = getImagesAndLabels('dataSet')
                recognizer.train(faces, np.array(Ids))
                recognizer.save('trainner/trainner.yml')
                sampleNum = sampleNum + 1
        except:
            FoundId = "?"
            cv2.imwrite("dataSet/User." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            print("New " + "dataSet/User." + Id + '.' + str(sampleNum) + ".jpg")
            faces, Ids = getImagesAndLabels('dataSet')
            recognizer.train(faces, np.array(Ids))
            recognizer.save('trainner/trainner.yml')
            sampleNum = sampleNum + 1

        select(10, x, y, t, (255, 255, 255), Id, image, 1)

        # incrementing sample number

        # saving the captured face in the dataset folder

    cv2.imshow('VIDEO', image)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > first + 10:
        break
camera.release()
cv2.destroyAllWindows()