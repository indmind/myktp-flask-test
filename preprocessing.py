import numpy as np
import cv2

def order_points(pts):
    # Inisialisasi matriks nol untuk menyimpan 4 titik sudut hasil pengurutan
    rect = np.zeros((4, 2), dtype="float32")

    # Menghitung jumlah sumbu x dan y dari setiap titik
    s = pts.sum(axis=1)

    # Menyimpan titik dengan nilai minimum dan maksimum dari jumlah sumbu
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Menghitung perbedaan sumbu x dan y antar titik
    diff = np.diff(pts, axis=1)

    # Menyimpan titik dengan nilai minimum dan maksimum dari perbedaan sumbu
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    ratio = 85.6 / 53.98

    if maxWidth > maxHeight:
        maxHeight = int(maxWidth / ratio)
    else:
        maxWidth = int(maxHeight * ratio)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def remove_glare(image, iteration=1):
    # Menghapus kilau dari gambar menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    result = image
    for i in range(iteration):
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Mengubah ukuran gambar sesuai dengan lebar atau tinggi yang ditentukan
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
