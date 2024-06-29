import cv2
import matplotlib.pyplot as plt
from fine_tune_kidney_stone_data.helper_functions import get_ground_truth_rect_list
import glob
import os
import random


if __name__=='__main__':
    root_path = ""  ## Working Directory

    data_path = "yolov10_kidneystone/fine_tune_kidney_stone_data/kidney_stone_data_roboflow/train"
    imgage_list = glob.glob(os.path.join(root_path,data_path,"images","*.jpg"))

    fig, ax = plt.subplots(1,5,sharex=True,sharey=True)
    for i in range(5):
        image_path = imgage_list[random.randint(0, len(imgage_list))]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = image_path.replace(".jpg", ".txt")
        label_path = os.path.join(root_path, data_path, "labels", image_path.split("/")[-1].replace(".jpg", ".txt"))

        ground_truth_rect_list = get_ground_truth_rect_list(label_path,img_shape=image.shape)
        ax[i].imshow(image)
        for rect in ground_truth_rect_list:
            ax[i].add_patch(rect)

    plt.show()

