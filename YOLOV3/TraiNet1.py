import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MyDatasets1 import MyDataset
from net import Darknet53
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ' torch.argmax(output, dim=1)'
class Trainner:#定义训练类
    def __init__(self):
        self.batch_size = 12
        self.save_path = 'models/pet.pth'#实例化保存的地址
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断用cuda
        self.net = Darknet53().to(self.device)#实例化网络
        if os.path.exists(self.save_path):#判断是否有之前训练过的网络参数，有的话就加载参数接着训练
            self.net.load_state_dict(torch.load(self.save_path))
        self.traindata = MyDataset()#实例化制作的数据集
        self.trainloader = DataLoader(self.traindata, batch_size=self.batch_size, shuffle=True)#加载数据集
        self.conf_loss_fn = nn.BCEWithLogitsLoss()#定义自信度的损失函数，这里可以用bceloss，不过bceloss要用sinmoid函数激活，用这个bce损失函数不需要用其激活
        self.crood_loss_fn = nn.MSELoss()#定义偏移量的均方差损失函数
        self.cls_loss_fn = nn.CrossEntropyLoss()#定义多分类的交叉熵损失函数
        self.optimzer = optim.Adam(self.net.parameters())#定义网络优化器
        self.summaryWriter = SummaryWriter("logs")
        # if os.path.exists(self.summaryWriter):
        #     self

    def loss_fn(self,output,target,alpha):#定义损失函数，并传入三个参数，网络输出的数据，标签，和用来平衡正负样本损失侧重那个的权重
        # 网络输出的数据形状[N,C,H,W]-->>标签形状[N,H,W,C]
        output=output.permute(0, 2, 3, 1)#将形状转成和标签一样
        #通过reshape变换形状 [N,H,W,C]即[N,H,W,45]-->>[N,H,W,3,15]，分成3个建议框每个建议框有15个值
        output=output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask_obj=target[..., 0]>0#获取正样本   拿到置信度大于0的掩码值返回其索引，省略号代表前面不变在最后的15个值里面取第0个，第0个为置信度
        #target[...,0] N 13 13 3 15 获取最后一个维度的第一个索引
        # print(mask_obj.shape)
        # print(mask_obj)
        # print(target[...,0].shape)

        output_obj=output[mask_obj]#获取输出的正样本
        # print(output_obj.shape)
        target_obj=target[mask_obj]#获取标签的正样本
        # print(target_obj.shape)
        loss_obj_conf=self.conf_loss_fn(output_obj[:, 0], target_obj[:, 0])#算置信度损失
        loss_obj_crood=self.crood_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])#算偏移量损失
        loss_obj_cls=self.cls_loss_fn(output_obj[:, 5:], torch.argmax(target_obj[:, 5:], dim=1))#可以用这个也可以用下面这个,
        # 或用下面这个也可以，用下面这个不需要对标签取最大值所以，用上面这个要对标签取最大值所以，因为输出是标量
        # loss_obj_cls=self.conf_loss_fn(output_obj[:,5:],target_obj[:,5:])
        loss_obj=loss_obj_conf+loss_obj_crood+loss_obj_cls#正样本总的损失

        mask_noobj = target[..., 0] == 0#获取负样本的掩码
        output_noobj = output[mask_noobj]#根据掩码获取数据
        target_noobj = target[mask_noobj]#根据掩码获取标签
        # print(output_noobj)
        # print(target_noobj)
        loss_noobj=self.conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])#计算负样本损失，负样本自信度为0，因此这里差不多就是和0 做损失
        loss = alpha*loss_obj+(1-alpha)*loss_noobj  #这里权重调整正负样本训练程度，如果负样本多久可以将权重给大点，负样本少就可以把权重给小点
        return loss#将损失返回给调用方

    def train(self):#定义训练函数
        self.net.train()#这个可用可不用表示训练，网络如果用到batchnormal及dropout等，但测试里面必须添加self.net.eveal()
        # epochs = 1#训练批次
        j = 0
        for epoch in range(1, 300):
            sum_loss = 0.
            sum_loss_13 = 0.
            sum_loss_26 = 0.
            sum_loss_52 = 0.
            with tqdm(total=1891//self.batch_size + 1, desc=f"Epoch {epoch} / 10000", mininterval=1, ncols=100, colour="blue") as pbar:

                for i, (target_13, target_26, target_52, img_data) in enumerate(self.trainloader):#循环trainloader，将3个返回值分别复制给对应变量
                    target_13, target_26, target_52, img_data = target_13.to(self.device), target_26.to(self.device), target_52.to(self.device), img_data.to(self.device)#将数据和标签转入cuda
                    output_13, output_26, output_52 = self.net(img_data)#将数据传入网络获得输出
                    #print(output_13)
                    loss_13 = self.loss_fn(output_13, target_13, 0.9)#自信度损失
                    loss_26 = self.loss_fn(output_26, target_26, 0.9)#偏移量损失
                    loss_52 = self.loss_fn(output_52, target_52, 0.9)#分类损失

                    loss = loss_13+loss_26+loss_52#总损失
                    self.optimzer.zero_grad()#清空梯度
                    loss.backward()#反向求导
                    self.optimzer.step()#更新梯度
                    sum_loss += loss.item()
                    sum_loss_13 += loss_13.item()
                    sum_loss_26 += loss_26.item()
                    sum_loss_52 += loss_52.item()
                    # torch.save(self.net.state_dict(),self.save_path)
                    self.summaryWriter.add_scalar("avg_loss_13训练损失", loss_13, j)
                    self.summaryWriter.add_scalar("avg_loss_26训练损失", loss_26, j)
                    self.summaryWriter.add_scalar("avg_loss_52训练损失", loss_52, j)
                    if i % 50 == 0:
                        torch.save(self.net.state_dict(), self.save_path.format(epoch))#保存网络参数
                    # score = torch.mean(torch.eq(a, b).float())
                    # sum_score = sum_score + score
                    pbar.update(1)
                    j += 1
            avg_loss = sum_loss / len(self.trainloader)
            avg_loss_13 = sum_loss_13 / len(self.trainloader)
            avg_loss_26 = sum_loss_26 / len(self.trainloader)
            avg_loss_52 = sum_loss_52 / len(self.trainloader)

            print(f"--{epoch}轮次--")
            print(f"平均总损失为：{avg_loss}， 13平均损失：{avg_loss_13}， 26平均损失：{avg_loss_26}， 52平均损失：{avg_loss_52}")

            # epochs += 1


            time.sleep(0.2)
        self.summaryWriter.close()
if __name__ == '__main__':
    obj = Trainner()
    obj.train()
