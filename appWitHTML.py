from flask import Flask, request, render_template
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

import tensorflow
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D

from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

import cv2

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = ResNet50(weights= 'imagenet', include_top= False, input_shape=(224,224,3))
model.trainable = False
model = keras.Sequential([model, GlobalAveragePooling2D()])

# Load the precomputed vectors
with open('./vectors_lst.pkl', 'rb') as f:
    vectors = pickle.load(f)

# Load the precomputed file names
with open('./file_names.pkl', 'rb') as f:
    file_names = pickle.load(f)

# Dummy function to convert image to vector
# Replace with your actual model inference code
def image_to_vector(image_path):
  img = cv2.imread(image_path)
  img = cv2.resize(img, (224,224))
  img = np.array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)
  result = model.predict(img)
  result = result.flatten()
  result = result/norm(result)
  neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
  neighbors.fit(vectors)
  distances, indices = neighbors.kneighbors([result])
  return distances, indices


@app.route('/')
def index():
    return '''
    <h1>Visual Search</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert the uploaded image to vector
        distances, indices = image_to_vector(file_path)


        result_html = '<h2>Uploaded Image:</h2>'
        result_html += f'<img src="{file_path}" style="max-width: 300px;">'

        result_html += '<h2>Similar Images:</h2>'
        for img in indices[0][:]:
            result_html += f'<img src="static{file_names[img]}" style="max-width: 150px;">'

        return result_html

if __name__ == '__main__':
    app.run()
