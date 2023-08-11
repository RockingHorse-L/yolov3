# 建议框尺寸
IMG_HEIGHT = 416
IMG_WIDTH = 416
CLASS_NUM = 9

# anchor box 是对coco数据集聚类获得的建议框
ANCHORS_GROUP_KMEANS = {
    52: [[10, 13], [16, 30], [33, 23]],
    26: [[30, 61], [62, 45], [50, 119]],
    13: [[116, 90], [156, 198], [373, 326]]
}

# 自定义建议框
ANCHORS_GROUP = {
    13: [[63, 29], [55, 50], [69, 77]],
    26: [[42, 19], [45, 37], [33, 53]],
    52: [[13, 14], [30, 13], [24, 30]],
}

# 计算建议框的面积
ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]]
}

if __name__ == '__main__':
    for feature_size, anchors in ANCHORS_GROUP.items():
        print(feature_size)
        print(anchors)
    for feature_size, anchor_area in ANCHORS_GROUP_AREA.items():
        print(feature_size)
        print(anchor_area)