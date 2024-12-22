import cv2
import numpy as np
from matplotlib import pyplot as plt

from services import *

plt.rcParams['figure.figsize'] = [20, 10]

image = cv2.imread('s255.png')
image_context = ImageContext(image=image)
image_context.corners = ImageService.find_rectangle_corners(image_context)

digit_recognizer = DigitRecognizer("../data/model.h5")

transformed_image = ImageService.perspective_transform(image_context)
cells = ImageService.split_into_cells(transformed_image)

sudoku_input = digit_recognizer.recognize_digits(cells)
sudoku_result = SudokuSolver.solve(sudoku_input)

image_context.sudoku_result = sudoku_result
result_image = np.zeros([307, 307, 3], dtype=np.uint8)
result_image = ImageDrawService.draw_digits(result_image, image_context)

cv2.imshow('wr', result_image)
cv2.waitKey(0)
