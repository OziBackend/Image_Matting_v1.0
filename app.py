from flask import Flask, request, send_file, jsonify
import os
from PIL import Image
import numpy as np
import shutil
import io

app = Flask(__name__)

# Define paths
input_folder = 'demo/image_matting/colab/input'
output_folder = 'demo/image_matting/colab/output'
foreground_folder = 'demo/image_matting/colab/foreground'
pretrained_ckpt = 'pretrained/modnet_photographic_portrait_matting.ckpt'

# Ensure folders exist
for folder in [input_folder, output_folder, foreground_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(input_folder, file.filename)
    file.save(file_path)

    # Run inference
    os.system(f'python -m demo.image_matting.colab.inference '
              f'--input-path {input_folder} '
              f'--output-path {output_folder} '
              f'--ckpt-path {pretrained_ckpt}')

    # Load and process the image
    matte_name = file.filename.split('.')[0] + '.png'
    image = Image.open(file_path)
    matte = Image.open(os.path.join(output_folder, matte_name))
    rgba_foreground_image = generate_rgba_foreground(image, matte, file.filename)

    # Save the foreground image to a BytesIO object and return it
    img_byte_arr = io.BytesIO()
    rgba_foreground_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png', download_name='foreground.png')

def generate_rgba_foreground(image, matte, image_name):
    # Calculate display resolution
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255

    # Calculate the foreground with the original color
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

    # Convert the foreground to an RGBA image (with transparency)
    foreground_image = Image.fromarray(np.uint8(foreground)).convert('RGB')
    alpha_matte = Image.fromarray(np.uint8(matte[:, :, 0] * 255)).convert('L')
    rgba_foreground_image = Image.merge('RGBA', (foreground_image.split()[0], foreground_image.split()[1], foreground_image.split()[2], alpha_matte))
    
    # Save the foreground image
    foreground_folder = 'demo/image_matting/colab/foreground'
    os.makedirs(foreground_folder, exist_ok=True)
    foreground_path = os.path.join(foreground_folder, f'foreground_{image_name}')
    rgba_foreground_image.save(foreground_path, format='PNG')
    
    return rgba_foreground_image

if __name__ == '__main__':
    app.run(host='172.16.2.46', port=5000, debug=True)
