import multiprocessing
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from joblib import Parallel, delayed

ann_folder = r"C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3\masks_instances\2024-05-22-14-22-09-239741"
mask_folder = r"C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3\masks_instances"
output_folder = r"C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3\energy_ann"
index = 0

def create_energy_picture(folder):
    ann_folder = os.path.join(mask_folder, folder)
    for index_ann, ann_name in enumerate(os.listdir(ann_folder)):
        ann_file = os.path.join(ann_folder, ann_name)
        img_annotation = cv2.imread(ann_file, cv2.IMREAD_GRAYSCALE)
        img_annotation = img_annotation/255.

        distance_transformed_img = ndimage.distance_transform_edt(img_annotation)

        for index, value in enumerate(np.unique(distance_transformed_img)[1:nuances]):

            distance_transformed_img[distance_transformed_img == value] = index*(1/nuances)

        distance_transformed_img[distance_transformed_img > 1.] = 1.

        if index_ann == 0:
            full_img = np.copy(distance_transformed_img)
        else:
            full_img = sum([full_img, distance_transformed_img])

    # matplotlib.use("TkAgg")
    # plt.imshow((full_img*255).astype(int))
    # plt.show()

    cv2.imwrite(os.path.join(output_folder, folder + ".jpg"), (full_img*255).astype(int))

if __name__ == "__main__":
    nuances= 40
    num_cores = multiprocessing.cpu_count()
    output = Parallel(n_jobs=num_cores)(delayed(create_energy_picture)(folder) for folder in tqdm(tqdm(os.listdir(mask_folder))))





