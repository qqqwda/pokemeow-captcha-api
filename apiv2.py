from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from flask import abort
# Initialize Flask app
app = Flask(__name__)

# Force TensorFlow to use only the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# Load the model at startup
model = tf.saved_model.load('inference_graph_v2/saved_model')

def load_label_map():
    return {
        1: '0',
        2: '1',
        3: '2',
        4: '3',
        5: '4',
        6: '5',
        7: '6',
        8: '7',
        9: '8',
        10: '9',
        11: 's0',
        12: 's1',
        13: 's2',
        14: 's3',
        15: 's4',
        16: 's5',
        17: 's6',
        18: 's7',
        19: 's8',
        20: 's9'
    }

# Load your label map
category_index = load_label_map()

# Example secret key, replace with your actual RapidAPI Proxy Secret
RAPIDAPI_PROXY_SECRET = os.getenv('RAPIDAPI_PROXY_SECRET')

def require_rapidapi_proxy_secret(f):
    def wrapper(*args, **kwargs):
        proxy_secret = request.headers.get('X-RapidAPI-Proxy-Secret')
        if proxy_secret != RAPIDAPI_PROXY_SECRET:
            abort(403)  # Forbidden access if the secret doesn't match
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    
    if len(request.files) > 1:
        return jsonify({'error': 'only one file can be processed at a time'}), 400

        #Validate the image size and format is more than 100kb
    if request.content_length > 100000:
        return jsonify({'error': 'file size too large'}), 400


    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image_np = np.array(image).astype(np.uint8)

    output_dict = run_inference_for_single_image(model, image_np)
    
    # Process detection results
    detected_classes = output_dict['detection_classes']
    detected_boxes = output_dict['detection_boxes']
    detected_scores = output_dict['detection_scores']

    # Filter detections based on score and extract only the numbers
    detections = [(category_index.get(class_id, 'Unknown'), score, box) for class_id, score, box in zip(detected_classes, detected_scores, detected_boxes) if score > 0.82]

    # Sort detections based on the x-coordinate of the bounding box (from left to right)
    detections.sort(key=lambda x: x[2][1])

    # Extract only the numbers
    number = ''.join([det[0] for det in detections])
    number = number.replace('s', '')
    return jsonify({'number': number})

def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

if __name__ == '__main__':
    app.run(host='0.0.0.0')
