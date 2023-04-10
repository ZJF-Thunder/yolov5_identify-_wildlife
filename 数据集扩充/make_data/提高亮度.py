import cv2
import os
import numpy as np

address = "C://Users//Dilraba//Desktop//huang"  # 存放原图片文件夹路径
list = os.listdir(address)
for i, file in enumerate(list):
    firstname = os.path.splitext(file)[0]
    typename = os.path.splitext(file)[1]
    #os.rename("{}/{}".format(address, file), "{}/{}.jpg".format(address, i + 1))
    newpath = address + "/" + "{}.jpg".format(i + 1)

    cv2.namedWindow("Image")  # 创建窗口
    img = cv2.imread(newpath)
    img_t = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_t)
    # 增加图像亮度
    v1 = np.clip(cv2.add(1*v,80),0,255)
    img1 = np.uint8(cv2.merge((h, s, v1)))
    img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    changepic = "C://Users//Dilraba//Desktop//liu//{}.jpg".format(i + 1)  # 修改完质量的图片存放路径
    cv2.imwrite(changepic, img1)

cv2.waitKey(0)