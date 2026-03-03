import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from math import log10, sqrt

# ==============================
# 1. Load Citra
# ==============================
img_ref = cv2.imread("Lurus.jpeg")
img_test = cv2.imread("Miring.jpeg")

img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

h, w = img_ref.shape


# ==============================
# 2. Fungsi Evaluasi
# ==============================
def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse_val))


# ==============================
# 3. Translasi, Rotasi, Scaling
#    (Koordinat Homogen)
# ==============================
def transform_homogeneous(img):

    tx, ty = 50, 30
    angle = 15
    scale = 1.2

    angle_rad = np.deg2rad(angle)

    # Matriks Translasi
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])

    # Matriks Rotasi
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                  [np.sin(angle_rad),  np.cos(angle_rad), 0],
                  [0, 0, 1]])

    # Matriks Scaling
    S = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]])

    # Kombinasi Matriks
    M = T @ R @ S
    M = M[:2, :]

    result = cv2.warpAffine(img, M, (w, h))
    return result


# ==============================
# 4. Transformasi Affine
# ==============================
def transform_affine(img):

    pts1 = np.float32([[50,50],
                       [200,50],
                       [50,200]])

    pts2 = np.float32([[10,100],
                       [200,50],
                       [100,250]])

    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, M, (w, h))

    return result


# ==============================
# 5. Transformasi Perspektif
# ==============================
def transform_perspective(img):

    pts1 = np.float32([[100,100],
                       [400,100],
                       [100,400],
                       [400,400]])

    pts2 = np.float32([[80,120],
                       [420,80],
                       [100,420],
                       [400,450]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, M, (w, h))

    return result


# ==============================
# 6. Interpolasi
# ==============================
def apply_interpolation(img, M, method):

    if method == "nearest":
        interp = cv2.INTER_NEAREST
    elif method == "bilinear":
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_CUBIC

    return cv2.warpPerspective(img, M, (w, h), flags=interp)


# ==============================
# 7. Evaluasi Interpolasi
# ==============================
def evaluate_interpolation():

    pts1 = np.float32([[100,100],
                       [400,100],
                       [100,400],
                       [400,400]])

    pts2 = np.float32([[80,120],
                       [420,80],
                       [100,420],
                       [400,450]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    results = {}

    for method in ["nearest", "bilinear", "bicubic"]:

        start = time.time()
        result = apply_interpolation(img_ref, M, method)
        end = time.time()

        mse_val = mse(img_ref, result)
        psnr_val = psnr(img_ref, result)
        comp_time = end - start

        results[method] = (mse_val, psnr_val, comp_time)

    return results


# ==============================
# 8. Eksekusi Pipeline
# ==============================

homog = transform_homogeneous(img_ref)
affine = transform_affine(img_ref)
perspective = transform_perspective(img_ref)

results_interp = evaluate_interpolation()

print("=== HASIL EVALUASI INTERPOLASI ===")
for method, values in results_interp.items():
    print(f"{method} -> MSE: {values[0]:.2f}, PSNR: {values[1]:.2f}, Time: {values[2]:.5f}s")


# ==============================
# 9. Visualisasi
# ==============================

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(img_ref, cmap='gray')

plt.subplot(2,2,2)
plt.title("Homogeneous Transform")
plt.imshow(homog, cmap='gray')

plt.subplot(2,2,3)
plt.title("Affine")
plt.imshow(affine, cmap='gray')

plt.subplot(2,2,4)
plt.title("Perspective")
plt.imshow(perspective, cmap='gray')

plt.show()