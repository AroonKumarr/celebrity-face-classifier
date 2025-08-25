import os
import json
import joblib
import numpy as np
import base64
import cv2
from wavelet import w2d

# -------- GLOBALS --------
current_dir = os.path.dirname(os.path.abspath(__file__))
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

# Map your model's folder/class names to UI keys used in HTML and element IDs
# Adjust the left side keys to exactly match your dataset/class_dictionary.json keys.
UI_KEY_MAP = {
    "babar_azam": "babar",
    "hania_amir": "hania",
    "virat_kohli": "virat",
    "waqar_zaka": "waqar",
}

def _to_ui_key(name: str) -> str:
    return UI_KEY_MAP.get(name, name)

# -------- IMAGE CLASSIFICATION --------
def classify_image(image_base64_data=None, file_path=None):
    """
    Classify an image either from a base64 string or a file path.
    Returns list of dicts:
      {
        'class': <raw_class_name>,
        'ui_key': <short key for HTML>,
        'class_probability': [...],
        'class_dictionary': {raw_name: idx, ...},
        'class_dictionary_ui': {ui_key: idx, ...}
      }
    """
    if __model is None:
        raise RuntimeError("Model not loaded. Call load_saved_artifacts() first.")

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []

    for img in imgs:
        # Resize and preprocess
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scaled_raw_img.reshape(32*32*3, 1),
            scaled_img_har.reshape(32*32, 1)
        ))

        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)

        # Predict class
        predicted_class = __model.predict(final)[0]
        predicted_proba = np.around(__model.predict_proba(final) * 100, 2).tolist()[0]

        # Build a UI-mapped dictionary for the probability table ids in HTML
        class_dict_ui = {_to_ui_key(k): v for k, v in __class_name_to_number.items()}

        result.append({
            'class': predicted_class,
            'ui_key': _to_ui_key(predicted_class),
            'class_probability': predicted_proba,
            'class_dictionary': __class_name_to_number,
            'class_dictionary_ui': class_dict_ui
        })

    return result


def class_number_to_name(class_num: int) -> str:
    return __class_number_to_name.get(class_num, "Unknown")


# -------- MODEL LOADING --------
def load_saved_artifacts():
    global __class_name_to_number, __class_number_to_name, __model

    print("Loading saved artifacts...")
    class_dict_path = os.path.join(current_dir, 'artifacts', 'class_dictionary.json')
    model_path = os.path.join(current_dir, 'artifacts', 'saved_model.pkl')

    # Load class dictionary
    with open(class_dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load model once
    if __model is None:
        with open(model_path, 'rb') as f:
            __model = joblib.load(f)

    print("Artifacts loaded successfully.")


# -------- IMAGE PROCESSING HELPERS --------
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1] if "," in b64str else b64str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path=None, image_base64_data=None):
    """
    Detect faces and return cropped images containing at least 2 eyes.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    elif image_base64_data:
        img = get_cv2_image_from_base64_string(image_base64_data)
    else:
        return []

    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


# -------- TEST HELPER --------
def get_b64_test_image_for_virat():
    test_file = os.path.join(current_dir, "b64.txt")
    with open(test_file) as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(get_b64_test_image_for_virat(), None))
