import os
labels = os.listdir('fashion_mnist_images/train')
print(labels)

files = os.listdir('fashion_mnist_images/train/0')
print(files[:10])
print(len(files))

import cv2
image_data = cv2.imread('fashion_mnist_images/train/4/0011.png',
cv2.IMREAD_UNCHANGED)
# print(image_data)

import matplotlib.pyplot as plt
plt.imshow(image_data, cmap='gray')
plt.show()
