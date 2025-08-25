import cv2
import pywt
import numpy as np

def w2d(img, wavelet='db1', level=1, mode='symmetric'):
    # Convert to grayscale
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, wavelet=wavelet, mode=mode, level=level)

    # Zero out detail coefficients at all levels (keep only approximation)
    coeffs_H = list(coeffs)
    for i in range(1, len(coeffs_H)):
        cH, cV, cD = coeffs_H[i]
        coeffs_H[i] = (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))

    # Reconstruct image
    imArray_H = pywt.waverec2(coeffs_H, wavelet=wavelet, mode=mode)
    imArray_H = np.clip(imArray_H, 0, 255).astype(np.uint8)
    return imArray_H
