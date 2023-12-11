import base64
import cv2

def image_to_base64(image):
    return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

