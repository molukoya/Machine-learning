import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

img = cv2.imread('asguard2.png',0)
median = cv2.medianBlur(img,5)
plt.imshow(median, cmap = 'gray')
img_city = cv2.imread('city.png',0)

#rotation angle in degree
rotated = ndimage.rotate(img_city, 45)

dft = cv2.dft(np.float32(rotated),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.imshow(magnitude_spectrum, cmap = 'gray')
