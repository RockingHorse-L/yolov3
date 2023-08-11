import cv2

from net import *
import cfg
import torch
import numpy as np
import PIL.ImageDraw as draw
import tool1
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageFont,ImageDraw

device = "cuda" if torch.cuda.is_available() else "cpu"

class Detector(torch.nn.Module):#定义侦测模块

    def __init__(self,save_path):
        super(Detector, self).__init__()

        self.net = Darknet53().to(device)#实例化网络
        self.net.load_state_dict(torch.load(save_path, map_location=device))#加载网络参数
        self.net.eval()#固化参数即去掉batchnormal的作用

    def forward(self, input, thresh, anchors):#定义前向运算，并给三个参数分别是输入数据，自信度阀值，及建议框
        output_13, output_26, output_52 = self.net(input)#将数据传入网络并获得输出
        #（n,h,w,3,15）其中n,h,w,3做为索引，即这里的idxs_13，表示定义在那个格子上
        idxs_13, vecs_13 = self._filter(output_13, thresh)#赛选获取13*13特侦图自信度合格的自信度的索引和15个值，赛选函数下面自己定义的
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])#对上面输出的两个值进行解析，获得最终的输出框，解析函数在下面
        idxs_26, vecs_26 = self._filter(output_26, thresh)#赛选获取26*26特侦图自信度合格的自信度的索引和15个值，赛选函数下面自己定义的
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])#对上面输出的两个值进行解析，获得最终的输出框，解析函数在下面
        idxs_52, vecs_52 = self._filter(output_52, thresh)#赛选获取52*52特侦图自信度合格的自信度的索引和15个值，赛选函数下面自己定义的
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])#对上面输出的两个值进行解析，获得最终的输出框，解析函数在下面
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)#将三种框在0维度按顺序拼接在一起并返回给调用方

    def _filter(self, output, thresh):#赛选过滤自信度函数，将自信度合格留下来
        output = output.permute(0, 2, 3, 1) # 数据形状[N,C,H,W]-->>标签形状[N,H,W,C]，，因此这里通过换轴

        # 通过reshape变换形状 [N,H,W,C]即[N,H,W,45]-->>[N,H,W,3,15]，分成3个建议框每个建议框有15个值
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = output[..., 0] > thresh#获取输出自信度大于自信度阀值的目标值的掩码（即布尔值），
        idxs = mask.nonzero()#将索引取出来其形状（N,H,W,3）
        vecs = output[mask]#通过掩码获取对应的数据
        return idxs, vecs#将索引和数据返回给调用方


    def _parse(self, idxs, vecs, t, anchors):#定义解析函数，并给4个参数分别是上面筛选合格的框的索引，9个值（中心点偏移和框的偏移即类别数），
        # t是每个格子的大小，t=总图大小/特征图大小，anchors建议框
        anchors = torch.Tensor(anchors).to(device)#将建议框转为Tensor类型
        #idx形状（N，V）
        a = idxs[:, 3]#表示拿到3个框对应的索引
        confidence = torch.sigmoid(vecs[:, 0])#获取自信度vecs里面有5+类别数个元素，第一个为自信度，因此取所有的第0个,输出的置信度会大于1，这里可以用其压缩到0-1之间。

        _classify = vecs[:, 5:]#获取分类的类别数
        if len(_classify) == 0:#判断类别数的长的是否为0为0返回空，避免代码报错
            classify = torch.Tensor([]).to(device)
        else:
            classify = torch.argmax(_classify, dim=1).float()#如果不为0，返回类别最大值的索引，这个索引就代表类别
        # idx形状（n，h，w，3），vecs（iou，p_x，p_y，p_w，p_h，类别）这里p_x，p_y代表中心点偏移量,p_w，p_h框偏移量
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t#计算中心点cy（h+p_y)
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t#计算中心点cx（h+p_x)
        #a = idxs[:, 3]  # 表示拿到3个对应的索引
        w = anchors[a, 0] * torch.exp(vecs[:, 3])#计算实际框的宽为w,w=建议框的宽*框的偏移量p_w
        h = anchors[a, 1] * torch.exp(vecs[:, 4])#计算实际框的高为h,h=建议框的高*框的偏移量p_h
        x1 = cx - w / 2#计算框左上角x的坐标


        y1 = cy - h / 2#计算框左上角y的坐标
        x2 = x1 + w#计算框右角x的坐标
        y2 = y1 + h#计算框右下角y的坐标下
        out = torch.stack([confidence,x1,y1,x2,y2,classify], dim=1)#将自信度坐标和类别按照一轴即列的方向重组堆叠在一起
        return out

if __name__ == '__main__':
    print(device)
    save_path = "models/pet.pth"
    detector = Detector(save_path)#实例化侦测模块
    #mg_path = 'data/images/'
    video_capture = cv2.VideoCapture(r'D:\AIdata\yolov3_train\20221111_150115.mp4')
    #img_name_list = os.listdir(img_path)
    name = {0: '白花鱼', 1: '黑花鱼', 2: '金鱼', 3: '白胖鱼', 4: '黑胖鱼', 5: '金胖鱼', 6: '细长鱼', 7: '大鱼', 8: '小鱼'}
    color = {0: "red", 1: "orange", 2: "blue", 3: "green", 4: 'maroon', 5: 'gray', 6: 'yellow', 7: 'coral', 8: 'darkturquoise'}
    font = ImageFont.truetype("simsun.ttc", 7, encoding="unic")  # 设置字体
    # for image_file in img_name_list:
    #     im =  Image.open(os.path.join(img_path,image_file))

        # y = detector(torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP)
        # print(y.shape)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # 缩放比例k，>1表示放大，<1表示缩小
        scale_w = 416 / w
        scale_h = 416 / h
        width_size = int(w * scale_w)
        hidth_size = int(h * scale_h)

        frame = cv2.resize(frame, (width_size, hidth_size))

        # top = abs(416 - h) // 2
        # bottom = abs(416 - h - top)
        # left = abs(416 - w) // 2
        # right = abs(416 - w - left)
        # padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # frame = cv2.dnn.blobFromImage(padded_frame, 1, (416, 416), swapRB=True, crop=False)

        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # img1 = pimg.open(r'data\imageS\06.jpg')#打开图片
        # img = im.convert('RGB')#将类型转成rgb
        img = np.array(im) / 255#归一化
        img = torch.Tensor(img)#将数据转成Tensor
        img = img.unsqueeze(0)#图片的形状为(h,w,c)在0维度加一个轴变成(1，h，w，c)即（n，h，w，c）的形状
        img = img.permute(0, 3, 1, 2)#换轴将nhwc变成nchw
        img = img.to(device)

        out_value = detector(img, 0.8, cfg.ANCHORS_GROUP)#调用侦测函数并将数据，自信度阀值和建议框传入
        boxes = []#定义空列表用来装框

        for j in range(9):#循环判断类别数
            classify_mask = (out_value[..., -1] == j)#输出的类别如果和类别相同就做nms删掉iou的的框留下iou小的表示这不是同一个物体
            # ，如果不用这个可能会将不同类别因iou大于目标值而不被删除，因此这里做个判断，只在同一类别中做nms，这里是获取同个类别的掩码
            _boxes = out_value[classify_mask]#更具掩码索引对应的输出作为框
            boxes.append(tool1.nms(_boxes).to(device))#对同一类别做nms删掉不合格的框，并将框添加进列表



        for box in boxes:#3种特征图的框循环框torch.cat([boxes_13, boxes_26, boxes_52], dim=0),循环3次
            # print(i)
            img_draw = draw.ImageDraw(im)
            for i in range(len(box)):#循环单个特征图的框,循环框的个数次
                c,x1, y1, x2, y2,cls = box[i, :]#将自信度和坐标及类别分别解包出来
                # print(c,x1, y1, x2, y2)
                # print(int(cls.item()))
                # print(round(c.item(),4))#取值并保留小数点后4位
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=cls.item(), thickness=2)
                img_draw.rectangle((x1, y1, x2, y2), outline=color[int(cls.item())], width=2)
                img_draw.text((max(x1, 0) + 3, max(y1, 0) + 3), fill=color[int(cls.item())], text=str(int(cls.item())), font=font,width=2)
                img_draw.text((max(x1, 0) + 15, max(y1, 0) + 3), fill=color[int(cls.item())], text=name[int(cls.item())], font=font,width=2)
                img_draw.text((max(x1, 0) + 3, max(y1, 0) + 20), fill=color[int(cls.item())],text=str(round(c.item(),4)), font=font, width=2)
        cv2.imshow('video', cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyWindow()
        # im.save(os.path.join('Detector_results/results1_img/',image_file))
        # plt.clf()
        # plt.ion()
        # plt.axis('off')
        # plt.imshow(im)
        # plt.show()
        # plt.pause(3)
    # plt.close()
        # im.show()


'============================================================================================================='
'''
nonzero(a) 
nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。它的返回值是一个长度为a.ndim(数组a的轴数)的元组，元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。

（1）只有a中非零元素才会有索引值，那些零值元素没有索引值；

（2）返回的索引值数组是一个2维tuple数组，该tuple数组中包含一维的array数组。其中，一维array向量的个数与a的维数是一致的。

（3）索引值数组的每一个array均是从一个维度上来描述其索引值。比如，如果a是一个二维数组，则索引值数组有两个array，第一个array从行维度来描述索引值；第二个array从列维度来描述索引值。

（4）transpose(np.nonzero(x))函数能够描述出每一个非零元素在不同维度的索引值。

（5）通过a[nonzero(a)]得到所有a中的非零值
'''