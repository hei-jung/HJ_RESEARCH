import os
import cv2
import numpy as np
import pandas as pd

# zx_path = '../labels/bifurcation_zx_new.csv'
zx_path = '../labels/test_bifurcation_zx.csv'
df = pd.read_csv(zx_path, index_col=0)

# path = '/home/jhj/Desktop/labeling/mip_coronal/'
path = '/home/jhj/Desktop/labeling/test_mip_coronal/'

data = []

for i, index in enumerate(df.index):
    left_z, left_x = df.loc[index]['left_z'], df.loc[index]['left_x']
    right_z, right_x = df.loc[index]['right_z'], df.loc[index]['right_x']
    # print(left_z, left_x, right_z, right_x)
    data.append(min(left_z, right_z))

    # img_path = path + index + '.npy'
    # img = np.load(img_path)
    # img = cv2.merge((img, img, img))
    #
    # cv2.namedWindow(img_path)
    # cv2.moveWindow(img_path, 1600, 800)
    #
    # z = min(left_z, right_z)
    # cv2.circle(img, (z, left_x), 5, (0, 0, 255), -1)
    # cv2.circle(img, (z, right_x), 5, (0, 0, 255), -1)
    #
    # x = (left_x + right_x) // 2
    # cv2.circle(img, (z, x), 5, (0, 255, 255), -1)
    #
    # while (1):
    #     cv2.imshow(img_path, img)
    #     k = cv2.waitKey(20) & 0xFF
    #     if k == 27:
    #         cv2.destroyWindow(img_path)
    #         break

df['mid_z'] = data
print(df)
# df.to_csv('../labels/bifurcation_zx_mid_new.csv')
df.to_csv('../labels/test_bifurcation_zx_mid.csv')
