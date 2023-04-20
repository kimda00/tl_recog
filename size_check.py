import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
      print(maindir)
      for filename in file_name_list:
          apath = os.path.join(maindir, filename)
          ext = os.path.splitext(apath)[1]
          if ext in IMAGE_EXT:
              image_names.append(apath)
    return image_names
save_dir = './save'
image_path = './croped_tll/4'
image_list = get_image_list(image_path)
for images in image_list:

    fname = images.split('/')[-1].split('.')[0]
    
    print(images)
    img=cv2.imread(images)
    print(f'img shape is : {img.shape}')

    
    # 이미지 불러오기
    #img = cv2.imread('./croped_tll/4/1_000892.jpg')

    # 이미지 전처리

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_blur = cv2.GaussianBlur(img_hsv, (7, 7), 0)

    # 바운딩 박스 그리기
    contours, _ = cv2.findContours(img_blur[:,:,1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:
            x, y, w, h = cv2.boundingRect(cnt)
            print(f'bjsdckxf is : {x},{y},{w},{h}')
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 바운딩 박스 내부에서 HSV 색공간을 이용하여 특정 색상 검출
            img_roi = img_hsv[y:y+h, x:x+w]
            print(f'roi is : {img_roi.shape}')
            save_name = f'{save_dir}/{fname}.jpg'
            print(save_name)
            cv2.imwrite(save_name, img_roi)
            ratio = h/w
            print(ratio)
