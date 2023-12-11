import numpy as np
import cv2
import preprocessing

def localize_ktp(image):
    # Mengatur ukuran gambar menjadi lebar 800 piksel
    image = preprocessing.image_resize(image, width=800)

    # Menghapus kilau dari gambar
    clahe = preprocessing.remove_glare(image, 0)

    # Mengaburkan gambar dengan kernel 10x10
    blur = cv2.blur(clahe, (10, 10))

    # Thresholding untuk mendapatkan masker
    _, mask = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    # Erosi dan dilasi pada masker untuk membersihkan dan memperbesar area
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Deteksi tepi menggunakan metode Canny
    lt = 50
    edges = cv2.Canny(mask, lt, lt * 3)

    # Dilasi dan erosi pada tepi untuk menutup baris edge kartu
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=6)
    edges = cv2.erode(edges, kernel, iterations=5)

    # Menemukan kontur pada tepi gambar
    contours, img = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_contour = image.copy()

    # Mencari kontur dengan luas terbesar yang memiliki 4 titik
    max_area = 0
    best_rect = None

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_rect = approx

    # Menggambar kontur pada gambar asli dan kontur terbaik
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    cv2.drawContours(image_contour, contours, -1, (255, 0, 0), 2)
    cv2.drawContours(image_contour, [contours[max_index]], -1, (0, 0, 255), 2)

    # Melakukan transformasi perspektif jika kontur terdeteksi
    warped = image
    if best_rect is not None:
        cv2.drawContours(image_contour, [best_rect], -1, (0, 255, 0), 2)
        warped = preprocessing.four_point_transform(image, best_rect.reshape(4, 2))

    # Mengembalikan hasil proses
    return (mask, edges, image_contour, warped)

# Memuat Haar Cascade untuk deteksi wajah
haar_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

def localize_face(image):
    mn = [5, 5, 5, 2, 5, 3, 5, 5, 5, 5, 5]  # minNeighbors
    sc = [1.2, 1.2, 1.2, 1.2, 1.6, 1.1, 2, 1.2, 1.2, 1.5]  # scaleFactor

    faces_rects = haar_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

    face_segments = []

    for (x, y, w, h) in faces_rects:
        face_segment = image.copy()[y-20:y+h+20, x-20:x+w+20]

        facedet_img = cv2.rectangle(image.copy(), (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 5)

        face_segments.append(face_segment)

    return face_segments[0] if len(face_segments) > 0 else image
