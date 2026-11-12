import base64, json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
IMG = 64
model = tf.keras.models.load_model("model.keras")
labels = json.load(open("labels.json"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.get_json()["image"]
    img_bytes = base64.b64decode(data_url.split(",")[1])
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, (IMG, IMG)).astype("float32")
    frame = frame.reshape(1, IMG, IMG, 1)
    probs = model.predict(frame, verbose=0)[0]
    i = int(np.argmax(probs))
    return jsonify(label=labels[i], confidence=float(probs[i]))

if __name__ == "__main__":
    app.run(debug=True, port=5000)