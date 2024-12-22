import cv2
import numpy as np
from matplotlib import pyplot as plt

from services import *

plt.rcParams['figure.figsize'] = [20, 10]

image_context = ImageContext()
image_context.sudoku_result = [1, 2, 0, 0, 9, 0, 5, 8, 4, 4, 0, 0, 5, 0, 6, 0, 0, 8, 4, 0, 0, 9, 0, 0, 0, 4, 0, 0, 2, 8, 9, 1 , 3,
                      6, 1, 0 , 5, 8,2, 0, 0, 3, 0, 5, 8, 1, 4, 0, 0, 5, 0, 6, 0, 0, 5, 4, 0, 0, 9, 0, 8, 4, 0, 0, 3, 0,
                      0, 6, 0, 0, 8, 4, 0, 0, 0, 8, 4, 0, 2, 9, 0]
result_image = np.zeros([307, 307, 3], dtype=np.uint8)
result_image = ImageDrawService.draw_digits(result_image, image_context)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
cv2.imshow('wr', result_image)
cv2.waitKey(0)