import cv2
import numpy as np


Actors=['jacky_chan','will smith']

haar_cascade=cv2.CascadeClassifier("haarcascade_face.xml") #the detector face
features=np.load('features.npy',allow_pickle=True) #load features 
labels=np.load('labels.npy') #load labels  

face_recognizer=cv2.face.LBPHFaceRecognizer_create() #create object from the class cv2.face.LBPHFaceRecognizer
face_recognizer.read('trained.yml')

img=cv2.imread('test/f629763089.jpg')
im=cv2.resize(img,(680,480))
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',gray)

#detect face in the image
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)
for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]
    label,confidence = face_recognizer.predict(faces_roi)
    print(f'label ={Actors[label]} with a confidence {confidence}')
    cv2.putText(im,str(Actors[label]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1.0,(43, 57, 192),2)
    cv2.rectangle(im,(x,y),(x+w,y+h),(96, 67, 21),3)
cv2.imshow('recognized_face',im)
status = cv2.imwrite('C:/Users/Client/Desktop/face_recog/output/face_recognized.jpg', im)
print("[INFO] Image face_recognized.jpg written to filesystem: ", status)
cv2.waitKey(0)
cv2.destroyAllWindows()