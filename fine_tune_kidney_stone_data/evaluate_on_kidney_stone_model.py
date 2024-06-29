from ultralytics import YOLOv10
import random

print('Loading Model ')
checkpoint_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/run/detect/train/xxx.pt"
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')


print('Doing Prediction on a Natural Image')
import os,glob

root_path = "/Users/Girish/PycharmProjects"
data_path = "yolov10_kidneystone/kidney_stone_data_roboflow/test"
imgage_list = glob.glob(os.path.join(root_path, data_path, "images", "*.jpg"))

image_path = imgage_list[random.randint(0, len(imgage_list))]
result = model.predict(image_path)
print('Finish Prediction on Image')

print('Show Results ')
import matplotlib.pyplot as plt
plt.imshow(result[0].plot())
plt.show()
print('Done')

