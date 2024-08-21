import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

def prepare_image(img: Image.Image, target_size=(200, 200)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array