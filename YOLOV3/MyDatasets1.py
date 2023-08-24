import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math
import torch.nn.functional as F

# 标签数据地址
LABEL_FILE_PATH = r'D:\AIdata\yolov3-fishes\labels.txt'
IMG_BASE_DIR = r'D:\AIdata\yolov3-fishes\images'

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 做一个onehot分类函数
def one_hot(cls_num, i):
    # 编一个类别数的一维0数组
    b = np.zeros(cls_num)
    # 在指定的位置填充1
    b[i] = 1.
    return b
class MyDataset(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]

        strs = line.split()
        #print(strs[0])
        try:
            _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        except Exception as e:
            print(strs[0])
        img_data = transforms(_img_data)
        # 也可以用这种方式转成float型
        _boxes = np.array(list(map(float, strs[1:])))
        # 将 boxes表中的元素5等分并赋值给boxes
        # 这里为5的原因是因为每个框有5个标签，除以5就可以把每个框分开，拿到框的数量。
        boxes = np.split(_boxes, len(_boxes) // 5)

        # 循环标签框，并将三种建议狂，和
        # 每种输出的框分别负责给两个变量，循环的目的是输出有3中特征图，因此也需要三种标签
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM), dtype=np.float32)
            # shape 13 13 3 15
            # 在空的字典中以
            # feature_size为键形状为shape=(feature_size. feature_size. 3. 5 + cfe.cLASS_NUM))的0矩阵为值得字典，
            # feature_size, feature_size输出特征图尺寸大小 3为3个建议狂，5+cfgCLASSNUM自信度、两个中心点坐标，两个偏移量+类别数

            # 循环框的个数
            for box in boxes:
                # 将每个框的数据组成的列表解包赋值给对应变量
                cls, cx, cy, w, h = box
                # 计算中心点的在每个格子x向的偏移量和索引，
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                # 中心点x坐标乘以特征图的大小再除以原图大小，整数部分作为索引，小数部分作为偏移量，原本是cx(图片总的大小/特征图大小)
                # ，展开括号就等于cx乘以特征图大小除以图片总的大小。 y方向同理
                # mathmodf方法返回x整数部分与小数部分，两部分的数值符号与x同，整数部分以浮点型表示。
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)
                for i, anchor in enumerate(anchors):#循环3种建议框，并带索引分别赋值给ianchor
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    # 每个建议框的面积，面积计算在另一个模块
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # anchor[0]、anchor[1]为建议框放的宽和高，w / anchor[0]，h / anchor[1]代表建议框和是实际框在wh向的偏移量，并赋值
                    # 实际标签框的面积
                    p_area = w * h

                    # 计算建议框和实际框的iou用最小面积除以最大面积
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *F.one_hot(torch.tensor(int(cls)), num_classes=cfg.CLASS_NUM)]
                    )
                    # 根据索引将[iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *F.one_hot(torch.tensor(int(cls))]
                    # 填入对应的区域里面作为标签，表示该区域有目标物体，其他地方没有就为0，
                    # labels[feature_size] = np.zeros(shape=(feature size. feature size. 3. 5 + cfe.CLASS NUM)),
                    # 由于labels的形状使这种，因此要将5+cfp.CLASSNUM填入对应位置，因此要在3个框的内部的对应格子里面，
                    # 因此feature_size, feature_size，3要作为索引去索引对应位置，feature_size，feature_size代表特征图大小，
                    # 也代表格子数，因此针对一个框的目标位置索引应该为目标所在格子的位置和所在框的序号。
                    # 因此这里才会写成labels[feature_size][int(cy_index), int(cx_index),i]
        # 将三种标签和数据返回给调用方
        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    x = one_hot(4, 2)
    print(x)
    data = MyDataset()
    dataloader = DataLoader(data, 3, shuffle=True)
    for i, (target_13, target_26, target_52, img_data) in enumerate(dataloader):
        print(target_13.shape)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)