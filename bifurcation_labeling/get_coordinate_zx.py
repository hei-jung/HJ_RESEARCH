import os
import cv2
import numpy as np
import pandas as pd

# csv_path = './labels/bifurcation_zx_new.csv'
csv_path = './labels/test_bifurcation_zx.csv'
df = pd.DataFrame({'FOLDERNAME': [], 'left_z': [], 'left_x': [], 'right_z': [], 'right_x': []})
df.set_index('FOLDERNAME')


def get_coordinate(event, z, x, flags, params):
    global prevZ, prevX
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (z, x), 5, (0, 0, 255), -1)
        prevZ, prevX = params[1], params[2]
        params[1], params[2] = z, x
        foldername = params[0].split('.')[0]
        if prevZ != 0 and prevX != 0:
            df.loc[len(df.index)] = [foldername, prevZ, prevX, params[1], params[2]]
            print(df.loc[len(df.index) - 1])


# path = '/home/jhj/Desktop/labeling/mip_coronal/'
# img_path_list = os.listdir(path)

# new data
# csv = '/home/jhj/Desktop/data/2022_snu/admin_NeuroQuant_2018new.csv'
csv = './labels/test_data.csv'
label = pd.read_csv(csv, index_col=0)
img_path_list = [index + '.npy' for index in label.index]
# path = '/home/jhj/Desktop/labeling/mip_coronal/'
path = '/home/jhj/Desktop/labeling/test_mip_coronal/'

for i, img_path in enumerate(img_path_list):
    print(f"\n================= [{i + 1}]")
    img = np.load(path + img_path)
    img = cv2.merge((img, img, img))

    cv2.namedWindow(img_path)
    cv2.moveWindow(img_path, 1600, 800)

    currZ, currX = 0, 0
    cv2.setMouseCallback(img_path, get_coordinate, [img_path, currZ, currX])

    while (1):
        cv2.imshow(img_path, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(img_path)
            break

df.to_csv(csv_path, index=False)
