from flask import Flask, request, send_file, jsonify, url_for
import os
from PIL import Image
import numpy as np
import io
import threading

semaphores = threading.Semaphore(20)

from functions.image_matting import image_matting

app = Flask(__name__, static_folder='static')

# Configure Flask app
app.config['SERVER_NAME'] = '172.16.2.46:5000'
app.config['APPLICATION_ROOT'] = '/'
app.config['PREFERRED_URL_SCHEME'] = 'http'

# Define paths
input_folder = 'demo/image_matting/colab/input'
# Ensure folders exist
if not os.path.exists(input_folder):
    os.makedirs(input_folder)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(input_folder, file.filename)
    file.save(file_path)

    return_data = {}

    print("Acquiring semaphore")
    semaphores.acquire()

    t = threading.Thread(target=image_matting, args=(app, file_path, file, return_data))

    t.start()
    t.join()

    print(return_data)
    print("Releasing semaphore")
    semaphores.release()

    if not return_data:
        return jsonify({"error": "No data returned"}), 400

    return jsonify(return_data)

if __name__ == '__main__':
    app.run(host='172.16.2.46', port=5000, debug=True)
    # app.run(host='46.250.238.182', port=8008, debug=True)
