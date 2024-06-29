from ultralytics import YOLOv10
import random
from fine_tune_kidney_stone_data.visualize_original_data import get_ground_truth_rect_list
import os,glob
import cv2

def get_preidction_rect_list(prediction_result):
    rect_list = []
    p = prediction_result[0].boxes.xywh.detach().cpu().numpy()
    from matplotlib import patches
    for item in p:
        rect = patches.Rectangle((item[0]-item[2]/2, item[1]-item[3]/2), item[2], item[3], linewidth=1, edgecolor='r', facecolor='none')
        rect_list.append(rect)

    return rect_list

root_path = ""


print('Loading Model ')
checkpoint_path = os.path.join(root_path, "yolov10_kidneystone/runs/detect/train4/weights/best.pt")

model = YOLOv10(checkpoint_path)
print('Finish Loading Model')

data_path = "yolov10_kidneystone/kidney_stone_data_roboflow/train"
imgage_list = glob.glob(os.path.join(root_path, data_path, "images", "*.jpg"))

import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,6,sharex=True,sharey=True)
for i in range(6):
    image_path = imgage_list[random.randint(0, len(imgage_list))]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label_path = image_path.replace(".jpg", ".txt")
    label_path = os.path.join(root_path, data_path, "labels", image_path.split("/")[-1].replace(".jpg", ".txt"))

    ground_truth_rect_list = get_ground_truth_rect_list(label_path,img_shape=image.shape)

    ax[0][i].imshow(image)
    for rect in ground_truth_rect_list:
        ax[0][i].add_patch(rect)
    ax[0][i].set_title('Ground Truth')

    result = model.predict(image_path)
    prediction_rect_list = get_preidction_rect_list(result)

    ax[1][i].imshow(image)
    for rect in prediction_rect_list:
        ax[1][i].add_patch(rect)
    ax[1][i].set_title('Prediction')

plt.show()
print('Done')


