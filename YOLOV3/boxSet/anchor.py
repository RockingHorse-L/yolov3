# -*- coding: utf-8 -*-
import numpy as np
import random
import argparse
import os

# 参数名称
parser = argparse.ArgumentParser(description='使用该脚本生成YOLO-V3的anchor boxes\n')
path = r'D:\AIdata\yolov3-fishes\labels.txt'
out_path = r'D:\AIdata\yolov3-fishes\anchor.txt'
parser.add_argument('--input_num_anchors', default=6, type=int, help='9')
parser.add_argument('--input_cfg_width', type=int, help="416")
parser.add_argument('--input_cfg_height', type=int, help="416")
args = parser.parse_args()
'''
centroids 聚类点 尺寸是 numx2,类型是ndarray
annotation_array 其中之一的标注框
'''


def IOU(annotation_array, centroids):
    #
    similarities = []
    # 其中一个标注框
    w, h = annotation_array
    for centroid in centroids:
        c_w, c_h = centroid
        if c_w >= w and c_h >= h:  # 第1中情况
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:  # 第2中情况
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:  # 第3种情况
            similarity = c_w * h / (w * h + (c_h - h) * c_w)
        else:  # 第3种情况
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)
    # 将列表转换为ndarray
    return np.array(similarities, np.float32)  # 返回的是一维数组，尺寸为(num,)


'''
k_means:k均值聚类
annotations_array 所有的标注框的宽高，N个标注框，尺寸是Nx2,类型是ndarray
centroids 聚类点 尺寸是 numx2,类型是ndarray
'''


def k_means(annotations_array, centroids, eps=0.00005, iterations=200000):
    #
    N = annotations_array.shape[0]  # C=2
    num = centroids.shape[0]
    # 损失函数
    distance_sum_pre = -1
    assignments_pre = -1 * np.ones(N, dtype=np.int64)
    #
    iteration = 0
    # 循环处理
    while (True):
        #
        iteration += 1
        #
        distances = []
        # 循环计算每一个标注框与所有的聚类点的距离（IOU）
        for i in range(N):
            distance = 1 - IOU(annotations_array[i], centroids)
            distances.append(distance)
        # 列表转换成ndarray
        distances_array = np.array(distances, np.float32)  # 该ndarray的尺寸为 Nxnum
        # 找出每一个标注框到当前聚类点最近的点
        assignments = np.argmin(distances_array, axis=1)  # 计算每一行的最小值的位置索引
        # 计算距离的总和，相当于k均值聚类的损失函数
        distances_sum = np.sum(distances_array)
        # 计算新的聚类点
        centroid_sums = np.zeros(centroids.shape, np.float32)
        for i in range(N):
            centroid_sums[assignments[i]] += annotations_array[i]  # 计算属于每一聚类类别的和
        for j in range(num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))
        # 前后两次的距离变化
        diff = abs(distances_sum - distance_sum_pre)
        # 打印结果
        print("iteration: {},distance: {}, diff: {}, avg_IOU: {}\n".format(iteration, distances_sum, diff,
                                                                           np.sum(1 - distances_array) / (N * num)))
        # 三种情况跳出while循环：1：循环20000次，2：eps计算平均的距离很小 3：以上的情况
        if (assignments == assignments_pre).all():
            print("按照前后两次的得到的聚类结果是否相同结束循环\n")
            break
        if diff < eps:
            print("按照eps结束循环\n")
            break
        if iteration > iterations:
            print("按照迭代次数结束循环\n")
            break
        # 记录上一次迭代
        distance_sum_pre = distances_sum
        assignments_pre = assignments.copy()


if __name__ == '__main__':
    # 聚类点的个数，anchor boxes的个数
    num_clusters = args.input_num_anchors
    # 索引出文件夹中的每一个标注文件的名字(.txt)
    #names = os.listdir(args.input_annotation_txt_dir)
    # 标注的框的宽和高
    annotations_w_h = []

    # 读取txt文件中的每一行
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            w, h = line.split(' ')[4:6]  # 这时读到的w,h是字符串类型
            # eval()函数用来将字符串转换为数值型
            annotations_w_h.append((eval(w), eval(h)))
        f.close()
    # 将列表annotations_w_h转换为numpy中的array,尺寸是(N,2),N代表多少框
    annotations_array = np.array(annotations_w_h, dtype=np.float32)
    N = annotations_array.shape[0]
    # 对于k-means聚类，随机初始化聚类点
    random_indices = [random.randrange(N) for i in range(num_clusters)]  # 产生随机数
    centroids = annotations_array[random_indices]
    # k-means聚类
    k_means(annotations_array, centroids, 0.00005, 200000)
    # 对centroids按照宽排序，并写入文件
    widths = centroids[:, 0]
    sorted_indices = np.argsort(widths)
    anchors = centroids[sorted_indices]
    # 将anchor写入文件并保存
    f_anchors = open(out_path, 'w')
    #
    for anchor in anchors:
        f_anchors.write('%d,%d' % (int(anchor[0]), int(anchor[1])))
        f_anchors.write('\n')