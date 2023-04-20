import os
import cv2
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

BAG_EXT = [".png",".jpg"]
FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 

def get_bag_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in BAG_EXT:
                image_names.append(apath)
    return image_names

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

# 이미지 불러오기
# img = cv2.imread('./croped_tl/test_2.jpg')
def img_label_crop(img_list):
    for image in img_list:
        folder_name  = image.split('/')[-1].split('.')[0]
        save_img = './result1/'+ FOLDER_NAME +'/'+ folder_name
        make_dir(save_img)
        img = cv2.imread(image)
        height,width,_ = img.shape
        img_crop = [img[:,0:int(1*(width/4))],img[:,int(1*(width/4)):int(2*(width/4))],img[:,int(2*(width/4)):int(3*(width/4))],img[:,int(3*(width/4)):int(4*(width/4))]]

        ##double plot
        fig = plt.figure()
        ax = fig.add_subplot()

        id_list = []
        s_value_list = []
        for id, img in enumerate(img_crop):
            # 이미지 전처리
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_blur = cv2.GaussianBlur(img_hsv, (7, 7), 0)
            i_height,i_width,_  = img_blur.shape
            img_crop_again = img_blur[int(2*i_height/4):int(3*i_height/4),int(2*i_width/4):int(3*i_width/4)]
            # img_crop_again = img_blur

            # 바운딩 박스 그리기
            contours, _ = cv2.findContours(img_blur[:,:,2], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(img_blur[:,:,2], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            id_list.append([id])
            
            ###double plot
            img_roi = img_crop_again
            lower_color = np.array([0, 0, 0])
            upper_color = np.array([255, 255, 255])
            mask = cv2.inRange(img_roi, lower_color, upper_color)

            # 검출된 색상의 3D 좌표를 그리기
            indices = np.where(mask == 255)
            pixel_values = img_roi[indices]
            pixel_values_norm = pixel_values / 255.0
            s_value_list.append(pixel_values_norm[:,1])

            cv2.imwrite(save_img + f'/{id}_img.jpg',img_crop_again)

        id_l = np.array([x*len(s_value_list[i]) for i,x in enumerate(id_list)])
        s_value_list = (np.array(s_value_list))
        print(id_l.shape)
        print(s_value_list.shape)

        for id in range(len(s_value_list)):
            ax.scatter(id_l[id], s_value_list[id])

            plt.savefig(save_img + f'/{id}_plot.jpg',dpi=300)


if __name__ =='__main__':
    img_list = get_bag_list('./croped_tl/crop_classes')
    img_label_crop(img_list)

