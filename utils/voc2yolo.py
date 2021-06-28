"""
VOC 格式的数据集转化为 YOLO 格式的数据集
--root_path 输入根路径
"""
import os
import glob
import random
import xml.etree.ElementTree as ET

currentRoot = os.getcwd()
classes = ["alive_fish", "dead_fish"]  # 这里改成你自己的类名
train_percent = 0.8


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlfile, labelfile):
    with open(xmlfile, 'r') as in_file:
        with open(labelfile, 'w') as out_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    currentRoot = os.getcwd()
    imgdirpath = os.path.join(currentRoot, "images", "*")
    imgdirlist = glob.glob(imgdirpath)
    for i in range(len(imgdirlist)):
        imgdir = imgdirlist[i]
        labeldir = imgdir.replace('images', 'labels')
        if (not os.path.exists(labeldir)):
            os.mkdir(labeldir)
    imgdirpath = os.path.join(imgdirpath, "*.jpg")
    imgpathlist = glob.glob(imgdirpath)
    for i in range(len(imgpathlist)):
        imgfilepath = imgpathlist[i]
        labelfilepath = imgfilepath.replace('images', 'labels')
        labelfilepath = labelfilepath.replace('.jpg', '.txt')
        if (not os.path.exists(labelfilepath)):
            xmlfilepath = imgfilepath.replace('images', 'Annotations')
            xmlfilepath = xmlfilepath.replace('.jpg', '.xml')
            if (not os.path.exists(xmlfilepath)):
                print("no xml file exists: " + xmlfilepath)
                continue
            else:
                convert_annotation(xmlfilepath, labelfilepath)
    labelfilepath = os.path.join(os.path.join(currentRoot, "labels", "*", "*.txt"))
    labelfilepathlist = glob.glob(labelfilepath)
    imagelist = []
    for i in range(len(labelfilepathlist)):
        labelfilepath = labelfilepathlist[i]
        image = labelfilepath.replace('labels', 'images')
        image = image.replace('.txt', '.jpg')
        imagelist.append(image)
    print(len(imagelist))
    num = len(imagelist)
    trainnum = int(num * train_percent)
    random.shuffle(imagelist)
    print(len(imagelist))
    for i in range(num):
        if (i < trainnum):
            with open('train.txt', 'a') as trainf:
                trainf.write(imagelist[i] + '\n')
        else:
            with open('test.txt', 'a') as testf:
                testf.write(imagelist[i] + '\n')
