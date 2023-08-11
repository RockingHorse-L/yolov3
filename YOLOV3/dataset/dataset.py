import math
import os.path

import torch.nn.functional as F
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import cfg

LABEL_FILE_PATH = r'D:\AIdata\yolov3-fishes\labels.txt'
IMG_BASE_DIR = r'D:\AIdata\yolov3-fishes\images'

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1
    return b

class yolov3_Dataset(Dataset):
    def __init__(self):
        # 先读取文件里的信息
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    # 迭代序列 （数组，元组，字符串）
    def __getitem__(self, index):
        labels = {}
        # 根据index取出对应位置的图片数据
        line = self.dataset[index]
        # 将数据分开
        strs = line.split()
        # 打开图片
        # print(strs[0])
        path = os.path.join(IMG_BASE_DIR, strs[0])
        img_datas = Image.open(path)
        img_data = transforms(img_datas)
        # 取出坐标的两种方式，切片速度要快一些
        # all_boxes = np.array(float(x) for x in strs[1:])

        # map 将指定的函数，依次作用于可迭代对象的每个元素，并返回一个迭代器对象。
        # map(function,iterable,...)
        # function-我们指定的函数（或数据类型），可以是python内置的，也可以是自定义的。
        # iterable-可迭代的对象，例如列表，元组，字符串等。
        all_boxes = np.array(list(map(float, strs[1:])))
        # 因为每个框有5个标签
        boxes = np.split(all_boxes, len(all_boxes) // 5)

        # 循环标签框
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            # 根据feature_size作为key(13, 26, 52),和大小(achors)给字典创建相同大小
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM), dtype=np.float32)
            for box in boxes:
                cls, cx, cy, w, h = box
                # 取出中心点所在框的索引和偏移量
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                # 循环取出3个建议框
                for i, anchor in enumerate(anchors):
                    # 建议框的面积
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # 实际标签的面积
                    p_area = w * h

                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    # 将数据存放在对应的字典里
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *F.one_hot(torch.tensor(int(cls)), num_classes=cfg.CLASS_NUM)]
                    )

        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    x = one_hot(4, 2)
    print(x)
    data = yolov3_Dataset()
    dataloader = DataLoader(data, 3, shuffle=True)
    for i, (target_13, target_26, target_52, img_data) in enumerate(dataloader):
        print(img_data.shape)
        print(target_13.shape)

