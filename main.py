from flask import Flask, request
import cv2
import numpy as np
import localization
import helper

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/scan", methods=['POST'])
def scan():
    image = request.files['image']
    image_bytes = np.fromfile(image, np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # lokalisasi ktp
    (_, _, _, ktp) = localization.localize_ktp(image)
    # lokalisasi wajah
    face = localization.localize_face(ktp)

    # convert image to base64
    card_image = helper.image_to_base64(ktp)
    photo_image = helper.image_to_base64(face)

    card_data = {
        "nik": "1234567890",
        "nama": "Bima Bayu",
        "tempat_lahir": "Blitar",
        "tanggal_lahir": "01-01-1990",
        "jenis_kelamin": "Laki-laki",
        # todo: add more data
    }

    return {
        "card_image": card_image,
        "photo_image": photo_image,
        "card_data": card_data
    }
