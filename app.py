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

    image.save('uploads/uploaded_image.jpg')
    
    if os.path.getsize('uploads/uploaded_image.jpg') > 10485760:
        return 'Image size exceeds 10MB', 400

    image = tf.keras.utils.load_img('uploads/uploaded_image.jpg', target_size=(480, 480))
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = tf.image.resize(input_arr,(480, 480))
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
        print("No predictions made")
        return jsonify({'predictions':[], 'scores':[]}), 200

    top_values, top_indices = tf.nn.top_k(predictions[0], k=k)
    top_classes = [class_names[idx] for idx in top_indices]
    top_scores = [float(score[idx]) for idx in top_indices]

    print(f"Top {k} predictions: ")
    for i in range(k):
        print(top_classes[i], top_scores[i])

    os.remove('uploads/uploaded_image.jpg')

    return jsonify({'predictions':top_classes, 'scores': top_scores}) , 200

if __name__ == '__main__':
    with open("model/cnames.txt", "r") as file:
        class_names = eval(file.read())
    print("Loading main model...")
    model = tf.keras.models.load_model("model/model.h5")
    app.run(debug=True)