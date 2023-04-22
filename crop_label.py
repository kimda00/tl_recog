import os
import cv2
import time
from IoU import iou_yolo

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

def get_info_bbox(bbox):
    label_name = bbox[0]
    x_mid = bbox[1]
    y_mid = bbox[2]
    w = bbox[3]
    h = bbox[4]
    
    return label_name, x_mid, y_mid, w, h 

def get_tl_label(txt):
    bbox_list = []
    with open(txt, 'r') as f:
        label_lines = f.readlines()
        for line in label_lines:
            # 라벨 정보 파싱
            bbox = list(map(float, line.strip().split(' ')))
            #print(bbox)
            if bbox[0] not in [4, 6, 8, 9, 10, 11, 12, 13, 14]:
                pass
            else:
                ### xmid,ymid,weith, height
                label_name, x_mid, y_mid, w, h  = get_info_bbox(bbox)
                #print(x_mid, y_mid, w, h)
                bbox_list.append([label_name, x_mid, y_mid, w, h])
                #print(x_min, y_min, x_max, y_max)
                
    return bbox_list


def tl_label_crop(image,txt):
    print(image)
    bbox_list = get_tl_label(txt)#len sum of lines

    img = cv2.imread(image)
    height, width, _ = img.shape
    crop_img_list = []
   
    for ii in range(len(bbox_list)):
        iou_source, box = iou_yolo(bbox_list[ii-1][1:],bbox_list[ii][1:],width,height)
        print(iou_source)
        if iou_source > 0.5:
            continue
        label_name = bbox_list[ii][0]
        x_min, y_min, x_max, y_max = box

        crop_img_list.append([label_name, img[y_min:y_max, x_min:x_max]])
        
    return crop_img_list

def main(img_list,txt_list):
    output_path = './result'
    make_dir(output_path)
    for id, image in enumerate(img_list):
        fname = image.split('/')[-1].split('.')[0]
        save_dir = output_path + '/' + FOLDER_NAME + '/' + fname
        make_dir(save_dir)
        txt = txt_list[id]

        crop_img_list = tl_label_crop(image,txt)
        count = 0
        for crop_img in crop_img_list:
            cv2.imwrite(save_dir + '/' + EPITON_LIST[int(crop_img[0])] + f'_{count}.jpg', crop_img[1]) 
            count+=1

if __name__ == '__main__':

    img_path = './images1/'
    txt_path = './labels/'
    img_list = get_image_list(img_path)
    txt_list = get_txt_list(txt_path)

    main(img_list, txt_list)
