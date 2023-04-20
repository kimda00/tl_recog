import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from crop_label import img_label_crop

# img -> detect -> box -> class -> correction -> correted class
#                           saturation: threshold ->logic gate combination -> correted class
#                           bbox: label(class-> correted class)
#
# 
# 평가용  img -> detect -> box -> class -> correction ->
#                                  class <-correted class
#                                   true case 즉 같은 case
#                                   false case 즉 다른 case
#
#

############ XINGYOU AREA ############     
EPITON_LIST = [ 'person', 'bicycle','car','Motorcycle','green3_h', 'bus', 'red3_h','truck','yellow3_h','green4_h', ' red4_h',' yellow4_h',' redgreen4_h','redyellow4_h',' greenarrow4_h', 'red_v','yellow_v','green_v']
FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 
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

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)
############ XINGYOU AREA ############    


def process_img(img,save_path):
    # img = cv2.imread('./croped_tl/test_2.jpg')

    # 이미지 전처리
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_blur = cv2.GaussianBlur(img_hsv, (7, 7), 0)

    # 바운딩 박스 그리기
    contours, _ = cv2.findContours(img_blur[:,:,2],cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    img_roi = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_roi = img_hsv[y:y+h, x:x+w]

            height,width,_ = img_roi.shape
            img_crop = [img_roi[:,0:int(1*(width/4))],img[:,int(1*(width/4)):int(2*(width/4))],img[:,int(2*(width/4)):int(3*(width/4))],img[:,int(3*(width/4)):int(4*(width/4))]]
            ##double plot

            fig = plt.figure()
            ax = fig.add_subplot()

            id_list = []
            h_value_list = []
            s_value_list = []
            v_value_list = []
            for id, img in enumerate(img_crop):
                # 이미지 전처리
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_blur = cv2.GaussianBlur(img_hsv, (7, 7), 0)
                i_height,i_width,_  = img_blur.shape
                img_crop_again = img_blur[int(2*i_height/4):int(3*i_height/4),int(2*i_width/4):int(3*i_width/4)]

                # 바운딩 박스 그리기
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
                h_value_list.append(pixel_values_norm[:,0])
                s_value_list.append(pixel_values_norm[:,1])
                v_value_list.append(pixel_values_norm[:,2])

                cv2.imwrite(save_path + f'/{id}_img.jpg',img_crop_again)

            id_h = np.array([x*len(h_value_list[i]) for i,x in enumerate(id_list)])
            id_s = np.array([x*len(s_value_list[i]) for i,x in enumerate(id_list)])
            id_v = np.array([x*len(v_value_list[i]) for i,x in enumerate(id_list)])
            h_value_list = (np.array(h_value_list))
            s_value_list = (np.array(s_value_list))
            v_value_list = (np.array(v_value_list))

            for id in range(len(h_value_list)):
                ax.scatter(id_h[id], h_value_list[id])
                plt.savefig(save_path + f'/{id}_h_plot.jpg',dpi=300)

#            for id in range(len(s_value_list)):
 #               ax1.scatter(id_s[id], s_value_list[id])
 #               plt.savefig(save_path + f'/{id}_s_plot.jpg',dpi=300)
#
#            for id in range(len(v_value_list)):
 ##               ax2.scatter(id_v[id], v_value_list[id])
          #      plt.savefig(save_path + f'/{id}_v_plot.jpg',dpi=300)

    
### 러프하게 찾아준 신호등 즉 bbox를 비비박스를 잘라준 이미지에서 신호등 박스를 색 칠한 이미지, 색칠 구역 잘라낸 이미지 저장 
def box(img_list, txt_list):
    output_path = './654321'
    make_dir(output_path)
    for id, image in enumerate(img_list):
        fname = image.split('/')[-1].split('.')[0]
        save_dir = output_path + '/' + FOLDER_NAME + '/' + fname
        make_dir(save_dir)
        txt = txt_list[id]
        count = 0
        crop_img_list = img_label_crop(image,txt)
        for crop_list in crop_img_list:
            label = crop_list[0]
            crop_img = crop_list[1]
            save_path = save_dir + '/' + EPITON_LIST[int(label)]
            make_dir(save_path)
            process_img(crop_img,save_path)
#            img, img_roi = process_img(crop_img,save_path)
#            cv2.imwrite(save_dir + '/' + EPITON_LIST[int(label)] + f'_img_{count}.jpg', img) 
#            cv2.imwrite(save_dir + '/' + EPITON_LIST[int(label)] + f'_roi_{count}.jpg', img_roi) 
            count+=1

if __name__ == '__main__':
    img_path = './images1/'
    txt_path = './labels/'
    img_list = get_image_list(img_path)
    txt_list = get_txt_list(txt_path)
    box(img_list, txt_list)