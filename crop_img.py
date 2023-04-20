import cv2
import shutil
import os
import matplotlib.pyplot as plt
from random import randint
import numpy as np


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
txt_EXT = [".txt"]

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

def get_txt_list(path):
    txt_names = []
    for maindir, subdir, file_name_list in os.walk(path):
      print(maindir)
      for filename in file_name_list:
          apath = os.path.join(maindir, filename)
          ext = os.path.splitext(apath)[1]
          if ext in txt_EXT:
              txt_names.append(apath)
    return txt_names

def get_croped_label(img_path, txt_path):
    labels_file_name = get_txt_list(txt_path)
    bbox_list = []
    for labels_file in labels_file_name:
        with open(labels_file, 'r') as f:
            label_lines = f.readlines()
            for line in label_lines:
                # 라벨 정보 파싱
                bbox = list(map(float, line.strip().split(' ')))
                #print(bbox)
                if bbox[0] not in [4, 6, 8, 9, 10, 11, 12, 13, 14]:
                    pass
                else:
                    coor_bbox = bbox[1:]
                    #print(coor_bbox)
                    x_min, y_min, x_max, y_max = map(float, coor_bbox)
                    bbox_list.append((x_min, y_min, x_max, y_max))
                    #print(x_min, y_min, x_max, y_max)
    # 이미지 파일 로드
    images_file_name = get_image_list(img_path)
    for image_file in images_file_name:
        img = cv2.imread(image_file)
        y_min = int(y_min*img.shape[1])
        y_max = int(y_max*img.shape[1])
        x_min = int(x_min*img.shape[0])
        x_max = int(x_max*img.shape[0])
        if img is not None:
            cv2.imshow(f"{image_file}.jpg", img)  # 확장자 추가하여 저장
            cv2.waitKey(0)
        else:
            print("Error: Image is empty")
        output_path = './result'
        #print(type(img))
        print(x_min)
        print(x_max)
        crop_img = img[y_min:y_max, x_min:x_max]
        file_name = os.path.basename(image_file)  # 파일 이름 추출
        file_path = os.path.join(output_path, file_name)  # 경로와 파일 이름 결합
        print(crop_img)
        try:
            cv2.imwrite(f"{file_path}.jpg", crop_img)  # 확장자 추가하여 저장
        except:
            print("Error: Image is empty")
        
    

if __name__ == '__main__':

    img_path = './images1/'
    txt_path = './labels/'
    what = get_croped_label(img_path, txt_path)
    # images_file_name = get_image_list(img_path)
    # for image_file in images_file_name:
    #     print(image_file)
    #     # image_name=image_file[10:]
    #     img = cv2.imread(image_file)
        
    #     # cv2.imshow('dd', img)
    #     if img is not None:
    #         cv2.imwrite('f"{file_path}.jpg"', img)  # 확장자 추가하여 저장
    #     else:
    #         print("Error: Image is empty")
    
    
    
   
  
    