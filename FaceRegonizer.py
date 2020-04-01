import cv2
import tensorflow as tf
import numpy as np
from mtcnn.mtcnn import MTCNN
import pickle

face_detector = MTCNN(min_face_size=50)

facenet = tf.keras.models.load_model('.\\models\\facenet.h5')
[model, names] = pickle.load(open('.\\models\\SVC_classifier.pkl', 'rb'))

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    faces = face_detector.detect_faces(frame)
    if len(faces) >= 1:
        for face in faces:
            [x,y,w,h] = face['box']
            x,y,w,h = abs(x), abs(y), abs(w), abs(h)
            face_box = tf.expand_dims(cv2.resize(frame[y:y+h, x:x+w]/255.0,(160,160)), axis=0)
            face_box = (face_box - np.mean(face_box))/np.std(face_box)
            faceEmbed = facenet.predict(face_box)
            prediction = model.predict(faceEmbed)
            name = names[prediction[0]]
            if max(model.predict_proba(faceEmbed)[0]) > 0.8:
                cv2.putText(frame, name ,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame, 'Unknown' ,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0), 2)
        
    cv2.imshow('Live feed', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
