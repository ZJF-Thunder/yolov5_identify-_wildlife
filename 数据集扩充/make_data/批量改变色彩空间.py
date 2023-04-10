import cv2
import os
address = "C://Users//Dilraba//Desktop//huang" # 存放原图片文件夹路径
list = os.listdir(address)
for i, file in enumerate(list):
    firstname = os.path.splitext(file)[0]
    typename = os.path.splitext(file)[1]
    #os.rename("{}/{}".format(address, file), "{}/{}.jpg".format(address, i+1))
    newpath = address + "/"+"{}.jpg".format(i+1)

    cv2.namedWindow("Image")  # 创建窗口
    img = cv2.imread(newpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", img)
    changepic = "C://Users//Dilraba//Desktop//zeng//{}.jpg".format(i+1) # 修改完质量的图片存放路径
    cv2.imwrite(changepic, img)

cv2.waitKey(0)