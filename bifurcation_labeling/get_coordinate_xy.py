import os
import cv2
import numpy as np
import pandas as pd
import nibabel as nib

# zx_path = '../labels/bifurcation_zx_mid_new.csv'
zx_path = '../labels/test_bifurcation_zx_mid.csv'
df = pd.read_csv(zx_path, index_col=0)

left_y_list = []
right_y_list = []


def get_coordinate(event, y, x, flags, params):
    global prevY, prevX
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (y, x), 5, (0, 0, 255), -1)
        prevY, prevX, midX = params[1:4]
        params[1], params[2] = y, x
        if prevY != 0 and prevX != 0:
            midY = (prevY + y) // 2
            cv2.circle(img, (midY, midX), 5, (0, 255, 255), -1)
            left_y_list.append(prevY)
            right_y_list.append(y)


# base = '/home/jhj/Desktop/labeling/temp/'
base = '../test_nifti/'

for i, index in enumerate(df.index):
    print(f"\n================= [{i + 1}]")
    z = df.loc[index]['mid_z']

    file_name = os.listdir(base + index)[0]
    img = nib.load(base + index + '/' + file_name)
    img = img.get_fdata()
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2], 1))
    img = (img - img.min()) / (img.max() - img.min())
    img2d = img[:, :, z, :]

    img = cv2.merge((img2d, img2d, img2d))

    cv2.namedWindow(index)
    cv2.moveWindow(index, 1600, 800)

    left_x, right_x = df.loc[index]['left_x'], df.loc[index]['right_x']
    cv2.line(img, (0, left_x), (img.shape[1], left_x), (0, 255, 255), 1)
    cv2.line(img, (0, right_x), (img.shape[1], right_x), (0, 255, 255), 1)

    mid_x = (left_x + right_x) // 2
    currY, currX = 0, 0
    cv2.setMouseCallback(index, get_coordinate, [index, currY, currX, mid_x])
    while (1):
        cv2.imshow(index, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(index)
            break

df['left_y'] = left_y_list
df['right_y'] = right_y_list
print(df)
# df.to_csv('../labels/bifurcation_zyx_new.csv')
df.to_csv('../labels/test_bifurcation_zyx.csv')
