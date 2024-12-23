import cv2
import numpy as np
from matplotlib import pyplot as plt

from services import *

plt.rcParams['figure.figsize'] = [20, 10]

# image = cv2.imread('s255.png')
image = cv2.imread('image_1.jpg')
image_context = ImageContext(image=image)
image_context.corners = ImageService.find_rectangle_corners(image_context)

digit_recognizer = DigitRecognizer("../data/model.h5")

transformed_image = ImageService.perspective_transform(image_context)
cv2.imshow('transformed_image', transformed_image)

transformed_image_1 = ImageService.image_correction(transformed_image)
cv2.imshow('transformed_image 1', transformed_image_1)

transformed_image_2 = ImageService.image_correction_2(transformed_image)
cv2.imshow('transformed_image 2', transformed_image_2)

transformed_image_3 = ColorBalancer.gray_world(transformed_image)
cv2.imshow('transformed_image 3', transformed_image_3)

transformed_image_4 = ColorBalancer.scale_by_max(transformed_image)
cv2.imshow('transformed_image 4', transformed_image_4)

transformed_image_5 = ColorBalancer.min_is_white(transformed_image)
cv2.imshow('transformed_image 5', transformed_image_5)

cells = ImageService.split_into_cells(transformed_image)

# sudoku_input = digit_recognizer.recognize_digits(cells)
# print(sudoku_input)
# sudoku_result = SudokuSolver.solve(sudoku_input)
#
# image_context.sudoku_result = sudoku_result
# result_image = np.zeros([307, 307, 3], dtype=np.uint8)
# result_image = ImageDrawService.draw_digits(result_image, image_context)
#
# cv2.imshow('wr', result_image)

cv2.waitKey(0)
