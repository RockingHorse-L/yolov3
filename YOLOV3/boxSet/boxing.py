import numpy as np

from dataset.kmeans import kmeans, avg_iou


def load_dataset_():
    path = r'D:\AIdata\yolov3-fishes\labels.txt'
    dataset = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    for item in lines:
        strs = item.split()
        boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(boxes, len(boxes) // 5)
        for obj in boxes:
            if np.float64(obj[4]) >= 5. and np.float64(obj[3]) >= 5.:
                dataset.append([np.float64(obj[3]), np.float64(obj[4])])
    return np.array(dataset)


if __name__ == '__main__':
    data = load_dataset_()
    out = kmeans(data, k=9)
    int_out = (out).astype(int)
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))  # 求平均iou, 越高说明选出来的框越好
    print("Boxes:\n {}-{}".format(out[:, 0], out[:, 1]))  # 得到w * 416, h * 416,因为yolo输入是416
    # 9个框选出来后,要按照面积从小到大进行排序
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))  # 宽高比不应过大
    print(f'\n {int_out[np.argsort(out[:, 0] * out[:, 1])]}')