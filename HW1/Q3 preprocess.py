import os
import cv2
import numpy as np
from pathlib import Path


src = Path("C:/Users/User/Desktop/EKG_seg")
dst_img = Path("C:/Users/User/Desktop/U-net/VOCdevkit/VOC2007/JPEGImages")
dst_label = Path("C:/Users/User/Desktop/U-net/VOCdevkit/VOC2007/SegmentationClass")


for folder_name in os.listdir(src):
    folder_path = Path(src).joinpath(folder_name).as_posix()
    if os.path.isdir(folder_path):
        print(folder_path)
        img_path = Path(folder_path).joinpath("img.png").as_posix()
        label_path = Path(folder_path).joinpath("label.png").as_posix()

        img_png = cv2.imread(img_path)
        img_save_path = Path(dst_img).joinpath(folder_name+".jpg").as_posix()
        cv2.imwrite(img_save_path, img_png)

        label_png = cv2.imread(label_path)
        gray = cv2.cvtColor(label_png, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        red_coords = np.where(gray == 38)
        for i in range(len(red_coords[0])):
            x = red_coords[0][i]
            y = red_coords[1][i]
            mask[x][y] = 1

        green_coords = np.where(gray == 75)
        for i in range(len(green_coords[0])):
            x = green_coords[0][i]
            y = green_coords[1][i]
            mask[x][y] = 2

        label_save_path = Path(dst_label).joinpath(folder_name+".png").as_posix()
        cv2.imwrite(label_save_path, mask)