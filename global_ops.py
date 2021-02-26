import cv2
import numpy as np
# prefix_path = f'components\mini_gozar_numbers'
prefix_path = f'assets'
img = cv2.imread(f'{prefix_path}\img.png')
bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY_INV, bw)
contours = []
contours, _ = cv2.findContours(bw.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
x_scale =  0.32591093117408860323886639676114
y_scale = 0.34298245614035038721804511278196
counter = 0
for contour in contours:
    bound_rect = cv2.boundingRect(contour)
    digit = bw[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]].copy()
    digit_resized = digit.copy()
    h, w = digit.shape
    new_size = (int(w * x_scale), int(h * y_scale))
    digit_resized = cv2.resize(digit, new_size)
    img_png_arr = [~digit_resized, ~digit_resized, ~digit_resized, digit_resized]
    img_png = digit_resized.copy()
    img_png = cv2.merge(img_png_arr, img_png)
    cv2.imwrite(f'{prefix_path}\{str(counter)}-.png', img_png)
    counter = counter + 1
    cv2.imshow('', ~digit_resized)
    cv2.waitKey()

cv2.imshow('', bw)
cv2.waitKey()