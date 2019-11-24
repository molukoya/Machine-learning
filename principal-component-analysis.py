#Dependencies
import glob
import numpy as np
import matplotlib.pyplot as plt

### Principal Component Analysis
images=[]
for i in glob.glob('orl_faces/s1/**.pgm'):
    images.append(plt.imread(i))
    image_axis12 = np.mean(images, axis=(1,2))
    image_mean = np.mean(images)
    zero_mean_image = image_axis12-image_mean
plt.plot(zero_mean_image,'ro')
