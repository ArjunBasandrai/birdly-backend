from flask import Flask, jsonify, request
from flask_cors import CORS 
from time import sleep
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
app.secret_key="t34s"

with open("model/cnames.txt", "r") as file:
    class_names = eval(file.read())
print("Loading main model...", flush=True)
model = tf.keras.models.load_model("model/model.h5")
print("Model loaded successfully", flush=True)

@app.route('/', methods=['POST'])
def main():
    return 'Server is running', 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image provided', 400

    image_file = request.files['image']

    if image_file.filename == '' or not image_file:
        return 'No image provided', 400
    
    blob_data = image_file.read()    
    image = Image.open(io.BytesIO(blob_data))
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_size = image_bytes.tell()
    
    if image_size > 10485760:
        return 'Image size exceeds 10MB', 400

    image = image.resize((480, 480))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  
    predictions = model.predict(input_arr, verbose = 0)

    score = tf.nn.softmax(predictions[0])
    max_score = float(np.max(score))

    if max_score > 0.75:
        k = 5
    elif max_score > 0.5:
        k = 3
    elif max_score > 0.25:
        k = 1
    else:
        print("No predictions made", flush=True)
        return jsonify({'predictions':[], 'scores':[]}), 200

    top_values, top_indices = tf.nn.top_k(predictions[0], k=k)
    top_classes = [class_names[idx] for idx in top_indices]
    top_scores = [float(score[idx]) for idx in top_indices]

    print(f"Top {k} predictions: ", flush=True)
    for i in range(k):
        print(top_classes[i], top_scores[i], flush=True)

    return jsonify({'predictions':top_classes, 'scores': top_scores}) , 200

if __name__ == '__main__':
    app.run(debug=True)