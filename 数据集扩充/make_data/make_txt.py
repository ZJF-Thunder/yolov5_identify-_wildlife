"""
将图片数据分成训练集和测试集，将名称写在txt文件上
"""
import os
import random
trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = 'data/all_images'
txtsavepath = 'data/ImageSets'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv) #从所有list中返回tv个数量的项目
train = random.sample(trainval, tr)
if not os.path.exists('data/ImageSets/'):
    os.makedirs('data/ImageSets/')
ftrainval = open('data/ImageSets/trainval.txt', 'w', encoding='utf-8')
ftest = open('data/ImageSets/test.txt', 'w', encoding='utf-8')
ftrain = open('data/ImageSets/train.txt', 'w', encoding='utf-8')
fval = open('data/ImageSets/val.txt', 'w', encoding='utf-8')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()