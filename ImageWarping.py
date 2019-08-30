import cv2
import numpy as np
import math

img = cv2.imread('../road6/520_ipm.png')
rows, cols = img.shape[0], img.shape[1]
img_output = np.zeros((rows, cols, 3), dtype=img.dtype)

for i in range(rows):
    for j in range(cols):
        offset_x = 0
        offset_y = int(135.0 * math.sin(2 * 3.14 * -j / (2*rows)))
        if i+offset_y > 0:
            img_output[i,j] = img[(i+offset_y) % rows, j]
        else:
            img_output[i,j] = 0


cv2.imshow('Concave', img_output)
cv2.waitKey()
cv2.imwrite('../road6/520ipm_warp.jpg',img_output)