import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


def convert_to_base64(image, resize=None, max_size=512):
    # Check if the image is a numpy array
    if isinstance(image, np.ndarray):
        # Convert the numpy array to a PIL Image
        image = Image.fromarray(image)

    # Resize the image if a new size is specified
    if resize:
        if isinstance(resize, tuple) and len(resize) == 2:
            image = image.resize(resize, Image.LANCZOS)
        else:
            raise ValueError("Resize parameter must be a tuple of (width, height)")

    # Scale the image to fit within max_size while preserving aspect ratio
    if max_size:
        if isinstance(max_size, int) and max_size > 0:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        else:
            raise ValueError("Max size must be a positive integer")

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def resize_base64(base64_string, resize=None, max_size=512):
    # Decode the base64 string to image
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))

    # Resize the image if a new size is specified
    if resize:
        if isinstance(resize, tuple) and len(resize) == 2:
            image = image.resize(resize, Image.LANCZOS)
        else:
            raise ValueError("Resize parameter must be a tuple of (width, height)")

    # Scale the image to fit within max_size while preserving aspect ratio
    if max_size:
        if isinstance(max_size, int) and max_size > 0:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        else:
            raise ValueError("Max size must be a positive integer")

    # Convert the resized image back to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    resized_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return resized_base64
