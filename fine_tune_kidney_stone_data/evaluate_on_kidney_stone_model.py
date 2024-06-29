from ultralytics import YOLOv10
import random
from fine_tune_kidney_stone_data.visualize_original_data import get_ground_truth_image

print('Loading Model ')
checkpoint_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/runs/detect/train/xxx.pt"
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')


import os,glob

root_path = "/Users/Girish/PycharmProjects"
data_path = "yolov10_kidneystone/kidney_stone_data_roboflow/test"
imgage_list = glob.glob(os.path.join(root_path, data_path, "images", "*.jpg"))

import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,4,sharex=True,sharey=True)
for i in range(4):
    image_path = imgage_list[random.randint(0, len(imgage_list))]
    image,ground_truth_rect_list = get_ground_truth_image(image_path)
    result = model.predict(image_path)
    prediction  = result[0].plot()

    ax[0][i].imshow(image)
    for rect in ground_truth_rect_list:
        ax[0][i].add_patch(rect)
    ax[1][i].imshow(prediction)

    ax[0][i].set_title('Ground Truth')
    ax[1][i].set_title('Prediction')

plt.show()
print('Done')

