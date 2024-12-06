import rembg
from PIL import Image

def remove_background(image_path):
    input_path = image_path
    output_path = 'output.png'

    with open(input_path, 'rb') as f:
        input_image = Image.open(f)
        output_image = rembg.remove(input_image)

    output_image.save(output_path, 'PNG')
    return output_path

if __name__ == "__main__":
    image_path = 'Nature_Before.png'
    remove_background(image_path)

# ============================================


def remove_background(app, file_path, file, return_data):
    with app.app_context():
        try:
            print('try statement')
            input_image = Image.open(file_path)

            #Removing the Background
            output_image = rembg.remove(input_image)
        except BaseException as e:
            print('except statement')