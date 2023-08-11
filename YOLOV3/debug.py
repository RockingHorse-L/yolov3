import os.path

import cv2
import numpy as np

txt_path = r'C:\Program Files\feiq\Recv Files\labels.txt'
img_path = r'C:\Program Files\feiq\Recv Files\images'

with open(txt_path, encoding='utf-8') as f:
    file = f.readlines()
for item in file:
    # print(item)
    strs = item.split()
    # print(strs)
    img_name = strs[0]
    boxes = np.array(list(map(float, strs[1:])))
    # print(boxes)
    boxes = np.split(boxes, len(boxes) // 5)
    #print(boxes)
    img = cv2.imread(os.path.join(img_path, img_name))
    for data in boxes:
        print(img_name)
        label, x, y, w, h = data
        print(x, y, w, h)
        cv2.rectangle(img, (int(x - w // 2), int(y - h // 2)), (int(w // 2 + x), int(h // 2 + y)), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()
cv2.destroyWindow()
