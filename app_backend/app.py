from flask import Flask
from routes import bp
import utils
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(bp)

if __name__ == "__main__":
    print("Starting Python flask server for celebrity image classification")
    # Make sure uploads dir exists under static
    os.makedirs(os.path.join(os.path.dirname(__file__), "static", "uploads"), exist_ok=True)
    utils.load_saved_artifacts()
    app.run(port=5000, debug=True)
