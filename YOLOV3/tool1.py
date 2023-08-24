import numpy as np
import torch


def ious(box, boxes, isMin = False):#定义iou函数
    box_area = (box[3] - box[1]) * (box[4] - box[2])#计算自信度最大框的面积
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])#计算其他所有框的面积
    xx1 = torch.max(box[1], boxes[:, 1])#计算交集左上角x的坐标其他同理
    yy1 = torch.max(box[2], boxes[:, 2])
    xx2 = torch.min(box[3], boxes[:, 3])
    yy2 = torch.min(box[4], boxes[:, 4])

    # w = torch.max(0, xx2 - xx1)
    # h = torch.max(0, yy2 - yy1)#获取最大值也可以用下面这种方法
    w = torch.clamp(xx2 - xx1, min=0)#获取最大值
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h#计算交集面积

    # ovr1 = inter/torch.min(box_area, area)
    ovr2 = inter/ (box_area + area - inter)#交集面积/并集面积
    # ovr = torch.max(ovr2,ovr1)

    # if isMin:#用于判断是交集/并集，还是交集/最小面积（用于处理大框套小框的情况）
    #
    #     ovr = inter / torch.min(box_area, area)
    # else:
    #     ovr = inter / (box_area + area - inter)

    return ovr2

def nms(boxes, thresh=0.5, isMin = True):#定义nms函数并传3个参数，分别是框，自信度阀值，是否最小面积

    if boxes.shape[0] == 0:#获取框的个是看是否为0，为0没框就返回一个空的数组防止代码报错
        return torch.Tensor([])

    _boxes = boxes[(-boxes[:, 0]).argsort()]#对框进行排序按自信度从大到小的顺序
    r_boxes = []#定义一个空的列表用来装合格的框

    while _boxes.shape[0] > 1:#循环框的个数
        a_box = _boxes[0]#取出第一个（自信度最大的框）框最为目标框与 其他框做iou
        b_boxes = _boxes[1:]#取出剩下的所有框

        r_boxes.append(a_box)#将第一个框添加到列表

        # print(iou(a_box, b_boxes))

        index = torch.where(ious(a_box, b_boxes,isMin) < thresh)#对框做iou将满足iou阀值条件的框留下并反回其索引
        _boxes = b_boxes[index]#根据索引取框并赋值给_boxes，使其覆盖原来的_boxes

    if _boxes.shape[0] > 0:#判断是否剩下最后一个框
        r_boxes.append(_boxes[0])#将最后一个框，说明这是不同物体，并将其放进列表

    return torch.stack(r_boxes)

if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # bs = np.array([[1,1,10,10],[11,11,20,20]])
    # print(iou(a,bs))

    bs = torch.tensor([[1, 1, 10, 10, 40,8], [1, 1, 9, 9, 10,9], [9, 8, 13, 20, 15,3], [6, 11, 18, 17, 13,2]])
    # print(bs[:,3].argsort())
    print(nms(bs))
