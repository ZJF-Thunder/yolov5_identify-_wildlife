import cv2
import xml.etree.ElementTree as ET
import os,sys
from albumentations import  HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, \
    RandomRotate90, Transpose, ShiftScaleRotate, Blur, CenterCrop, RandomCrop, \
    OpticalDistortion, GridDistortion, HueSaturationValue, \
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, \
    IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, \
    RandomBrightness, Flip, OneOf, VerticalFlip, Resize, Rotate, Compose
import numpy as np


def pretty_xml(element, indent = '\t', newline = '\n', level=0):
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def insert_object(root, xmin, xmax, ymin, ymax):
    obj = ET.Element('object')
    obj.tail = '\n'
    root.append(obj)
    name = ET.Element('name')
    name.text = 'Bait'
    name.tail = '\n'
    obj.append(name)
    pose = ET.Element('pose')
    pose.text = 'Unspecified'
    pose.tail = '\n'
    obj.append(pose)
    truncated = ET.Element('truncated')
    truncated.text = '0'
    truncated.tail = '\n'
    obj.append(truncated)
    difficult = ET.Element('difficult')
    difficult.text = '0'
    difficult.tail = '\n'
    obj.append(difficult)

    bndbox = ET.Element('bndbox')
    bndbox.tail = '\n'
    obj.append(bndbox)
    x_min = ET.Element('xmin')
    x_min.text = str(xmin)
    x_min.tail = '\n'
    bndbox.append(x_min)
    y_min = ET.Element('ymin')
    y_min.text = str(ymin)
    y_min.tail = '\n'
    bndbox.append(y_min)
    x_max = ET.Element('xmax')
    x_max.text = str(xmax)
    x_max.tail = '\n'
    bndbox.append(x_max)
    y_max = ET.Element('ymax')
    y_max.text = str(ymax)
    y_max.tail = '\n'
    bndbox.append(y_max)

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    cv2.imshow('data_augmentation', img)
    cv2.waitKey(0)


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})


category_id_to_name = {0: 'Bait', 1: 'bait'}
aug_ver = get_aug([VerticalFlip(p = 1)])  #垂直方向翻转
aug_hor = get_aug([HorizontalFlip(p=1)])  #水平方向翻转
aug_res = get_aug([Resize(p=1, height=256, width=256)]) #resize
aug_cen = get_aug([CenterCrop(p=1, height=200, width=200)], min_area=4000)
aug_cen = get_aug([CenterCrop(p=1, height=100, width=100)], min_visibility=0.3) # 只返回变换后可见性大于 threshold 的 boxes
aug_ran = get_aug([RandomCrop(p=1, height=100, width=100)])
aug_SCR =get_aug([ShiftScaleRotate(shift_limit=0.0625,
                         scale_limit=1,
                         rotate_limit=45, p=1)])  #旋转、裁切
aug_rot = get_aug([Rotate(limit=60, p =1.0)])

aug_list = [aug_ver, aug_hor, aug_rot]  #想用哪个，就添加在找个list里

#--------------------------            读取xml，解析，增强图像，修改box信息，写入xml            -----------------------------#
if __name__ == '__main__':
    jpgPath = 'images'
    xmlPath = 'xml/'

    xmls = os.listdir(xmlPath)
    for xml in xmls:
        xmlName = xml.split('.')[0]
        imgName = xmlName + '.jpg'

        try:
            tree = ET.parse(os.path.join(xmlPath, xml))
            root = tree.getroot()
        except Exception as e:
            print('prase ' + xml + ' failed!')
            sys.exit()
        else:
            image = cv2.imread(os.path.join(jpgPath, imgName))
            for width in root.iter('width'):
                if int(width.text) == 0:
                    width.text = str(image.shape[1])
                    for height in root.iter('height'):
                        if int(height.text) == 0:
                            height.text = str(image.shape[0])
                            tree.write(os.path.join(xmlPath, xmlName + '.xml'))

            bboxes = []
            for object in root.findall('object'):
                for box in object.findall('bndbox'):
                    x_min = int(box.find('xmin').text)
                    x_max = int(box.find('xmax').text)
                    y_min = int(box.find('ymin').text)
                    y_max = int(box.find('ymax').text)
                    root.remove(object)
                bboxes.append([x_min, y_min, x_max, y_max])
            category_id = np.zeros(len(bboxes))
            annotations = {'image': image, 'bboxes': bboxes, 'category_id': category_id}

            for i, aug in enumerate(aug_list):
                aug_type = str(aug).split('(')[1][4:]
                augmented = aug(**annotations)

                for iter in range(len(augmented['bboxes'])):
                    x_min, y_min, x_max, y_max = augmented['bboxes'][iter]
                    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                    insert_object(root, x_min, x_max, y_min, y_max)

                for filename in root.iter('filename'):
                    name = filename.text.split('.')[0]
                    filename.text = name + aug_type + '.jpg'
                for path in root.iter('path'):
                    pathname = path.text.split('.')[0]
                    path.text = pathname + aug_type + '.jpg'
                for width in root.iter('width'):
                    width.text = str(image.shape[1])
                for height in root.iter('height'):
                    height.text = str(image.shape[0])

                if len(augmented['bboxes']) > 0:
                    cv2.imwrite(os.path.join(jpgPath, xmlName + aug_type +'.jpg'),  augmented['image'])
                    pretty_xml(root)
                    tree.write(os.path.join(xmlPath, xmlName + aug_type +'.xml'))
                    for object in root.findall('object'):
                        root.remove(object)

# #centerCrop
# aug = get_aug([CenterCrop(p=1, height=100, width=100)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# #certerCrop，并限定最小box面积
# aug = get_aug([CenterCrop(p=1, height=200, width=200)], min_area=4000)
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# # 只返回变换后可见性大于 threshold 的 boxes
# aug = get_aug([CenterCrop(p=1, height=100, width=100)], min_visibility=0.3)
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# aug = get_aug([RandomCrop(p=1, height=100, width=100)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# #旋转、裁切
# aug =get_aug([ShiftScaleRotate(shift_limit=0.0625,
#                          scale_limit=1,
#                          rotate_limit=45, p=1)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# #旋转
# aug = get_aug([Rotate(limit=60, p = 0.7)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

#多种增强混合，同时使用
# def augment_flips_color(p=.5):
#     return Compose([
#         # CLAHE(),
#         Transpose(),
#         ShiftScaleRotate(shift_limit=0.0625,
#                          scale_limit=1,
#                          rotate_limit=45, p=.75),
#         # Blur(blur_limit=3),
#         # OpticalDistortion(),
#         # GridDistortion(),
#         # HueSaturationValue()
#     ], p=p)
#
# aug = augment_flips_color(p=1)
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

