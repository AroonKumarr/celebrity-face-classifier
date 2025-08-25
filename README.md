# 🎯 Celebrity Face Classifier (Flask + Dropzone.js)

A web app that classifies uploaded celebrity photos using a pre-trained SVM model.  
Frontend: **Dropzone.js + Bootstrap** for smooth drag-and-drop uploads.  
Backend: **Flask** serving a prediction API that uses OpenCV + PyWavelets + scikit-learn.

---

## ✨ Features

- Drag & drop (or click) to select an image
- One-click **Classify** button
- Best-match **result card** + full **probability table**
- **Remove** the image and upload another without page refresh
- Robust face detection (face + 2 eyes) before classification

---

## 🧩 Architecture

- **`app.py`** – Flask app factory & server bootstrap
- **`routes.py`** – Blueprint with routes (`/`, `/classify_image`)
- **`utils.py`** – Loads artifacts, pre-processes images, runs predictions
- **`wavelet.py`** – Wavelet transform (`w2d`) feature extraction
- **`templates/app.html`** – UI (Bootstrap + Dropzone form + results table)
- **`static/app.js`** – Dropzone config + classification UI logic
- **`artifacts/`** – `saved_model.pkl`, `class_dictionary.json`
- **`static/`** – `app.css`, `dropzone.min.js`, `dropzone.min.css`, `images/`

---

## 📁 Project Structure

celebrity_face_classifier/
├─ app_backend/ # Flask backend + UI
│ ├─ app.py # Flask entrypoint
│ ├─ routes.py # API routes
│ ├─ utils.py # Model utilities
│ ├─ wavelet.py # Wavelet feature extractor
│ ├─ init.py
│ ├─ artifacts/ # Model files
│ │ ├─ saved_model.pkl
│ │ └─ class_dictionary.json
│ ├─ static/ # Frontend assets
│ │ ├─ app.css
│ │ ├─ app.js
│ │ ├─ dropzone.min.css
│ │ ├─ dropzone.min.js
│ │ ├─ images/ # Demo images
│ │ └─ uploads/ # Runtime upload folder
│ ├─ templates/
│ │ └─ app.html # Main UI
│ └─ b64.txt # (misc file, safe to ignore)
│
├─ dataset/ # Training dataset
│ ├─ babar_azam/
│ ├─ hania_amir/
│ ├─ virat_kohli/
│ ├─ waqar_zaka/
│ └─ cropped/ # Pre-processed faces
│
├─ ml_model/ # Training code
│ ├─ train_model.py # Script to train & save model
│ └─ openCV/ # OpenCV utils/config
│
├─ test_images/ # Sample images for testing
│ ├─ babar.jpeg
│ ├─ haina amir.avif
│ ├─ virat kohli.jpeg
│ └─ waqar zaka.jpg
│
├─ venv/ # Python virtual environment
├─ README.md # Project documentation
└─ Requirements.txt # Dependencies list
> ✅ Ensure the **`artifacts/`** folder contains the trained model and class dictionary before running.

---

## ⚙️ Setup

### 1) Create & activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
macOS / Linux

python -m venv venv
source venv/bin/activate
2) Install dependencies
Create requirements.txt (recommended minimal set):


Flask==3.1.1
numpy==2.0.2
opencv-python==4.11.0.86
scikit-learn==1.6.1
joblib==1.5.1
PyWavelets==1.6.0
Then install:


pip install -r requirements.txt
You may also have the following from experimentation (optional):
matplotlib, pandas, pillow. Avoid face-recognition/dlib unless you use them—those are heavy.

3) Place model artifacts
Put saved_model.pkl and class_dictionary.json into artifacts/.

Paths are read inside utils.py from artifacts/ relative to project root.

4) Create runtime folders

mkdir -p uploads


▶️ Run

python app.py
Open: http://127.0.0.1:5000

🖱️ How to Use
Page loads with a gallery of celebrities (cards).

Drag an image into the Dropzone or click to select.

Click “Classify”.

See the best match card and probability table.

Click Remove on the Dropzone to clear and upload a new image.

The UI won’t submit automatically—classification only runs when you click “Classify”.
You can remove the current image (Dropzone’s “Remove” link) and upload another without reloading the page.

🔌 API (Backend)
POST /classify_image
Content-Type: multipart/form-data

Field: file (image)

Response (example):


[
  {
    "class": "virat",
    "class_probability": [1.22, 96.45, 0.87, 1.46],
    "class_dictionary": {
      "babar": 0,
      "virat": 1,
      "hania": 2,
      "waqar": 3
    }
  }
]
🧠 Model Summary
Face detection: OpenCV Haar cascades (must find face + 2 eyes)

Features: concatenation of

resized RGB (32×32×3)

wavelet features using w2d (32×32)

Classifier: SVM (sklearn.svm.SVC) loaded from artifacts/saved_model.pkl

🐞 Troubleshooting
KeyError: 'file'
Ensure Dropzone is configured with:

paramName: "file"

You’re binding Dropzone to the correct form element (#dropzone)

autoProcessQueue: false and classification starts via dz.processQueue() when clicking Classify.

Static files 404
Make sure app.html uses url_for('static', filename='...') and the files exist under static/.

“Can’t classify image…” error
The detector needs a clear face with two visible eyes. Try better-lit, frontal images.

Artifacts path issues
Confirm artifacts/ exists and filenames match exactly.

🔐 Notes
This server is for development (Flask’s debug server). For production, use a WSGI server (e.g., Gunicorn/Uvicorn behind Nginx).



🤝 Contributing
PRs welcome! Please:

Keep endpoints backward compatible.

Update this README if you change UI or API.

Include screenshots for UI changes.

👤 Author  
**Aroon Kumar**  
- [GitHub: @AroonKumarr](https://github.com/AroonKumarr)  
- [LinkedIn: Aroon Kumar](https://www.linkedin.com/in/aroon-kumar)  
- 📱 Phone/WhatsApp: +92 331 03908377

📄 License
MIT © Aroon Kumar



If you’d like, I can also generate a ready-to-use `requirements.txt` and a `.gitignore` snippet for Python/Flask projects.