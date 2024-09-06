import os
from PIL import Image
from flask import jsonify, url_for
import numpy as np

output_folder = 'demo/image_matting/colab/output'
foreground_folder = 'static/foreground'  # Serve from static folder
pretrained_ckpt = 'pretrained/modnet_photographic_portrait_matting.ckpt'

for folder in [output_folder, foreground_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def image_matting(app, file_path, file, return_data):
    with app.app_context():
    # Run inference
        os.system(f'python -m demo.image_matting.colab.inference '
              f'--input-path {file_path} '
              f'--output-path {output_folder} '
              f'--ckpt-path {pretrained_ckpt}')

        # Load and process the image
        matte_name = file.filename.split('.')[0] + '.png'
        image = Image.open(file_path)
        matte = Image.open(os.path.join(output_folder, matte_name))
        rgba_foreground_image = generate_rgba_foreground(app, image, matte, file.filename)

        # Save the foreground image to the static folder
        foreground_image_name = f'foreground_{file.filename}'
        foreground_image_path = os.path.join(foreground_folder, foreground_image_name)
        rgba_foreground_image.save(foreground_image_path, format='PNG')

        # Generate the URL for the saved image
        image_url = url_for('static', filename=f'foreground/{foreground_image_name}', _external=True)

        return_data['image_url']={"image_url": image_url}


def generate_rgba_foreground(app, image, matte, image_name):
    with app.app_context():
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
        
        return rgba_foreground_image
