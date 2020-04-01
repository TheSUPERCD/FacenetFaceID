import os
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import pickle

detector = MTCNN(min_face_size=50)
model = load_model('.\\models\\facenet.h5')

face_dataset = {}
initial_name = os.listdir('.\\faces\\')[0].split('_')[0]
face_dataset[initial_name] = []
for filename in os.listdir('.\\faces\\'):
    name = filename.split('_')[0]
    img = cv2.imread('.\\faces\\' + filename, 1)
    print(filename)
    faces = detector.detect_faces(img)
    faces.sort(key=lambda x: abs(x['box'][2]*x['box'][3]))
    if len(faces) >=1 :
        x,y,w,h = abs(faces[0]['box'][0]), abs(faces[0]['box'][1]), abs(faces[0]['box'][2]), abs(faces[0]['box'][3])
        face_box = cv2.resize(img[y:y+h, x:x+w], (160,160))/255.0
        face_box = (face_box - np.mean(face_box))/np.std(face_box)
        if name == initial_name:
            face_dataset[name].append(face_box)
        else:
            face_dataset[initial_name] = model.predict(np.stack(face_dataset[initial_name], axis=0))

            initial_name = name
            face_dataset[initial_name] = []
            x,y,w,h = faces[0]['box']
            face_dataset[name].append(face_box)

face_dataset[name] = model.predict(np.stack(face_dataset[name], axis=0))

with open('.\\faceEmbeds\\face_embeds.pkl', 'wb+') as f:
    pickle.dump(face_dataset, f)
