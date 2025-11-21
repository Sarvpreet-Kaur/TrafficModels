# embedder.py
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# create a single global ResNet50 instance (include_top=False, pooling='avg' -> 2048-dim vector)
_resnet = None

def get_resnet():
    global _resnet
    if _resnet is None:
        _resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return _resnet

def image_to_embedding(pil_image, target_size=(224,224)):
    """
    pil_image: PIL.Image instance (RGB) or numpy array (H,W,3)
    returns: 1D numpy array (e.g., length 2048)
    """
    if not hasattr(pil_image, "resize"):
        pil_image = Image.fromarray(pil_image[..., ::-1])  # if BGR (cv2) -> convert to RGB
    img = pil_image.resize(target_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)          # ResNet50 preprocessing
    model = get_resnet()
    emb = model.predict(arr, verbose=0)  # shape (1, 2048)
    return emb.ravel()                   # shape (2048,)
