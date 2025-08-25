import numpy as np
import cv2
import matplotlib 
from matplotlib import pyplot as plt

img = cv2.imread('./test_images/Tom Cruise.jpeg')

print(img.shape)

plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(gray)
plt.imshow(gray,cmap = 'gray')

# this code is from opencv documation for detecting the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)
(x,y,w,h) = faces[0]
print(x,y,w,h)
# croping the part which we have to need:
face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#plt.imshow(face_img)

# this code we have take from opencv documation: for detecting the eyees

cv2.destroyAllWindows()
for (x, y, w, h) in faces:
    face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

#plt.figure()
#plt.imshow(face_img, cmap='gray')


#croping the image
plt.imshow(roi_color, cmap = 'gray')


# creating a function in which i will the input of image:
# then give back to me cropped image:

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

cropped_image = get_cropped_image_if_2_eyes('./test_images/aroon.jpeg')
plt.imshow(cropped_image)
#plt.show()

# created two paths one will get the image other create folder named:
# cropped:
import os
import shutil

# Directory paths
path_to_data = "./dataset/"
path_to_cr_data = './dataset/cropped/'

img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir() and entry.name != "cropped":
        img_dirs.append(entry.path)

print(img_dirs)

cropped_image_dirs = []
celebirty_file_names_dict = {}

# Fill celebirty_file_names_dict from existing cropped folders if skipped
for img_dir in img_dirs:
    celebirty_name = img_dir.split('/')[-1]
    cropped_folder = os.path.join(path_to_cr_data, celebirty_name)
    if os.path.exists(cropped_folder):
        file_list = [os.path.join(cropped_folder, f) for f in os.listdir(cropped_folder) if f.endswith('.png')]
        if len(file_list) > 0:
            celebirty_file_names_dict[celebirty_name] = file_list

    
    # ✅ Skip processing if cropped folder already exists and has images
    if os.path.exists(cropped_folder) and len(os.listdir(cropped_folder)) > 0:
        print(f"Skipping {celebirty_name}, cropped images already exist.")
        continue

    print("Processing:", celebirty_name)
    count = 0
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                celebirty_file_names_dict[celebirty_name] = []

            cropped_file_name = celebirty_name + str(count) + '.png'
            cropped_file_path = os.path.join(cropped_folder, cropped_file_name)

            cv2.imwrite(cropped_file_path, roi_color)
            celebirty_file_names_dict[celebirty_name].append(cropped_file_path)
            count += 1

      
            
            
            
import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    imArray = img

    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255;

    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H


im_har = w2d(cropped_image, 'db1', 5)
plt.imshow(im_har, cmap='gray')
#plt.show()

x = []
y = []

for celebirty_name, training_files in celebirty_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img,(32,32))
        img_har = w2d(img, 'db1',5)
        scalled_img_har = cv2.resize(img_har,(32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(celebirty_name)
        
print(len(x))

x = np.array(x).reshape(len(x),4096).astype(float)
print(x.shape)

print(x[0])

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


X_train,X_test,y_train,y_test = train_test_split(x,y, random_state = 0)
pipe = Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel = 'rbf',C = 10))])

pipe.fit(X_train,y_train)
score = pipe.score(X_test,y_test)
print(score)

print(len(X_test))

print(classification_report(y_test,pipe.predict(X_test)))

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

scores = []
best_estimators = {}
import pandas as pd

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
print(best_estimators)

import joblib
joblib.dump(best_estimators['svm'], 'saved_model.pkl')

import json
# Create class_dict mapping celebrity names to numeric labels
class_dict = {}
for idx, celeb in enumerate(celebirty_file_names_dict.keys()):
    class_dict[celeb] = idx

with open("class_dictionary.json",'w') as f:
    f.write(json.dumps(class_dict))
