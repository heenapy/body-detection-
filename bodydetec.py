import numpy as np
import cv2
import dlib
# ______________________________________________body______________________________________________--
body_classifier = cv2.CascadeClassifier('/home/paython/Desktop/haarcasecade/haarcascade_fullbody.xml')
cap = cv2.VideoCapture('/home/paython/A Kind Man Helps a Dog Cross the Road Viral Video!.mp4')
while cap.isOpened():
    ret,frame = cap.read()
    frame=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    for(x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('pedestrain', frame)
        if cv2.waitKey(1)==13:
            break
cap.release()
cv2.destroyAllWindows ()