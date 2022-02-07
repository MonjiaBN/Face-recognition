import os
import cv2 
import numpy as np


Actors=['jacky_chan','will_smith']
#or
'''Actors=[]
for actor in os.listdir('images/'):
    Actors.append(actor)
print(actor)'''

DIR=r'./images/'


features =[]  
labels=[]

#the Haar cascade object detection
haar_cascade=cv2.CascadeClassifier("haarcascade_face.xml")

def create_train():
    for actor in Actors:
        path=os.path.join(DIR,actor)   
        label=Actors.index(actor)

        for img in os.listdir(path):  
            img_path=os.path.join(path,img)   
            image=cv2.imread(img_path)      
            im=cv2.resize(image,(680,480))
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

            
            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8) #increase the minNeighbors to have a good result
            
            for (x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

print('-----------------Training done ------------------')

#train the recognizer on the features list and the labels list
features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer=cv2.face.LBPHFaceRecognizer_create() #local binary patterns histograms
face_recognizer.train(features,labels)
#store the histograms for each image
face_recognizer.save('trained.yml') 
np.save('features.npy',features,allow_pickle=True)
print("the array is saved in the file features.npy")

np.save('labels.npy',labels,allow_pickle=True)
print("the array is saved in the file labels.npy")


            
        

 
