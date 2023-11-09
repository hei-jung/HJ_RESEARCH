import pandas as pd
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('../labels/bifurcation_zyx.csv', index_col=0)
df = pd.read_csv('../labels/test_bifurcation_zyx.csv', index_col=0)

# base = '../input_nifti/'
base = '../test_nifti/'

# display image with dots
for i, index in enumerate(df.index):
    left_x, right_x = df.loc[index]['left_x'], df.loc[index]['right_x']
    y = (df.loc[index]['left_y'] + df.loc[index]['right_y']) // 2
    z = df.loc[index]['mid_z']

    file_name = os.listdir(base + index)[0]
    img = nib.load(base + index + '/' + file_name)
    img = img.get_fdata()
    img = (img - img.min()) / (img.max() - img.min())

    plt.subplot(1, 2, 1)
    plt.title('axial')
    plt.imshow(img[:, :, z], cmap='gray')
    plt.plot(y, left_x, 'ro')
    plt.plot(y, right_x, 'ro')

    plt.subplot(1, 2, 2)
    plt.title('coronal mip')
    plt.imshow(np.max(img, axis=1), cmap='gray')
    plt.plot(z, left_x, 'ro')
    plt.plot(z, right_x, 'ro')

    # # axial (z)
    # plt.subplot(1, 3, 1)
    # plt.title('axial mip')
    # plt.imshow(np.max(img, axis=2), cmap='gray')
    #
    # # coronal (y)
    # plt.subplot(1, 3, 2)
    # plt.title('coronal mip')
    # plt.imshow(np.max(img, axis=1), cmap='gray')
    # plt.plot(z, left_x, 'ro')
    # plt.plot(z, right_x, 'ro')
    #
    # # sagittal (x)
    # plt.subplot(1, 3, 3)
    # plt.title('sagittal mip')
    # plt.imshow(np.max(img, axis=0), cmap='gray')

    plt.show()
    plt.close()
    break
