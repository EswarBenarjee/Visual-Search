from flask import Flask, request, Response
from flask_cors import CORS  # Import CORS
import json
import os
import pickle
import numpy as np
import cv2  # Import OpenCV for image processing
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

import tensorflow
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D

from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

import cv2

app = Flask(__name__)

# Initialize CORS
CORS(app)

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
    <h1>Visual Search API</h1>
    <p>Use the /upload endpoint to upload an image and get similar images in JSON format.</p>
    '''

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        response = json.dumps({'error': 'No file part'})
        return Response(response, mimetype='application/json', status=400)
    
    file = request.files['file']
    if file.filename == '':
        response = json.dumps({'error': 'No selected file'})
        return Response(response, mimetype='application/json', status=400)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert the uploaded image to vector
        distances, indices = image_to_vector(file_path)

        # Prepare the JSON response
        response_data = {
            'uploaded_image': f'static/uploads/{filename}',
            'similar_images': []
        }

        for i in indices[0]:
            response_data['similar_images'].append({
                'image': f'static/{file_names[i]}',
                'distance': float(distances[0][list(indices[0]).index(i)])
            })

        response = json.dumps(response_data)
        return Response(response, mimetype='application/json')

if __name__ == '__main__':
    app.run(port=8000)  # Set the port to 8000
