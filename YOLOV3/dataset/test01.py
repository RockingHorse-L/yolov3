import glob
import xml.etree.cElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "D:\data\VOCdevkit\VOC2007\Annotations"
CLUSTERS = 9


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        try:
            for obj in tree.iter("object"):
                xmin = int(obj.findtext("bndbox/xmin")) / width  #简单做归一化
                ymin = int(obj.findtext("bndbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height

                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin:
                    print(xml_file)
                dataset.append([xmax - xmin, ymax - ymin])
        except:
            print(xml_file)
    return np.array(dataset)


if __name__ == '__main__':
    # print(__file__)
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    # clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    # out= np.array(clusters)/416.0
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))  #求平均iou, 越高说明选出来的框越好
    print("Boxes:\n {}-{}".format(out[:, 0] * 416, out[:, 1] * 416)) #得到w * 416, h * 416,因为yolo输入是416
    #9个框选出来后,要按照面积从小到大进行排序

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios))) #宽高比不应过大
