
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os
import random

def get_ground_truth_image(image_path,label_path=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if label_path is None:
        label_path = image_path.replace(".jpg",".txt")
        label_path = os.path.join(root_path,data_path,"labels",image_path.split("/")[-1].replace(".jpg",".txt"))

    with open(label_path, 'r') as file:
        labels = file.readlines()

    image_height, image_width, _ = image.shape

    rect_list = []
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.strip().split())
        class_name = ''

        # Convert YOLO format to bounding box coordinates
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x_min = x_center - width / 2
        y_min = y_center - height / 2

        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='g', facecolor='none')
        rect_list.append(rect)

    return image,rect_list




if __name__=='__main__':
    root_path = "/Users/Girish/PycharmProjects"
    data_path = "yolov10_kidneystone/kidney_stone_data_roboflow/test"
    imgage_list = glob.glob(os.path.join(root_path,data_path,"images","*.jpg"))

    fig, ax = plt.subplots(1,4)
    for i in range(4):
        image_path = imgage_list[random.randint(0, len(imgage_list))]
        image,rect_list = get_ground_truth_image(image_path)
        ax[i].imshow(image)
        for rect in rect_list:
            ax[i].add_patch(rect)
            # ax[i].text(rect.xy[0], rect.xy[1], '', color='white', fontsize=6, backgroundcolor='green')

    plt.show()

