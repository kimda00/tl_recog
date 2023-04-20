import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

# 이미지 불러오기
#img = cv2.imread('./croped_tll')

# 이미지 전처리

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_blur = cv2.GaussianBlur(img_hsv, (7, 7), 0)

# 바운딩 박스 그리기
contours, _ = cv2.findContours(img_blur[:,:,2], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 바운딩 박스 내부에서 HSV 색공간을 이용하여 특정 색상 검출
        img_roi = img_hsv[y:y+h, x:x+w]
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 255, 255])
        mask = cv2.inRange(img_roi, lower_color, upper_color)

        # 검출된 색상의 3D 좌표를 그리기
        indices = np.where(mask == 255)
        pixel_values = img_roi[indices]
        pixel_values_norm = pixel_values / 255.0
        # print(pixel_values_norm)
        x_label = 'H ({})'.format(pixel_values_norm[0, 0])
        y_label = 'S ({})'.format(pixel_values_norm[0, 1])
        z_label = 'V ({})'.format(pixel_values_norm[0, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pixel_values_norm[:, 0], pixel_values_norm[:, 1], pixel_values_norm[:, 2], c=pixel_values_norm)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.show()

# 결과 이미지 출력
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()