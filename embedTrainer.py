import tensorflow as tf
import pickle
import numpy as np
from sklearn.svm import SVC

with open('.\\faceEmbeds\\face_embeds.pkl', 'rb') as f:
    face_dataset = pickle.load(f)
    # print(face_dataset)

x_train = []
y_train = []
names = list(face_dataset)

for name in names:
    label = names.index(name)
    for arr in face_dataset[name]:
        x_train.append(arr)
        y_train.append(label)
x_train = np.stack(x_train, axis=0)
y_train = np.array(y_train)

model = SVC(kernel='linear', probability=True)
model.fit(x_train,y_train)
print(model.predict(x_train), model.predict_proba(x_train))
pickle.dump([model, names], open('.\\models\\SVC_classifier.pkl','wb+'))

# model = pickle.load(open('.\\models\\SVC_classifier.pkl', 'rb'))
# print(model.predict(x_train))