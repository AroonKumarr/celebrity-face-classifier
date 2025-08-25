import os
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import utils

bp = Blueprint("main", __name__)

# Always upload into app_backend/static/uploads so Flask can serve if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@bp.route("/")
def index():
    return render_template("app.html")

@bp.route("/classify_image", methods=["POST"])
def classify_image():
    """
    Receives an image file (form field name: 'file'),
    saves it temporarily, runs classification, returns JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded", "detail": "missing 'file' field"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    try:
        # IMPORTANT: use keyword to hit utils.classify_image(file_path=...)
        result = utils.classify_image(file_path=temp_path)

        # If no face with 2 eyes found, return empty list (frontend shows message)
        if not result:
            return jsonify([]), 200

        return jsonify(result), 200
    finally:
        # Best effort cleanup
        try:
            os.remove(temp_path)
        except Exception:
            pass
