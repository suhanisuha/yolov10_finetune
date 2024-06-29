from ultralytics import YOLOv10
import os
root_path = "" # deinfe your working directory path


print('Loading Model ')
checkpoint_path = os.path.join(root_path , "yolov10_kidneystone/yolov10_checkpoints/yolov10m.pt")
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')


print('Doing Prediction on a Natural Image')
# img_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/fine_tune_kidney_stone_data/Modi_with_Gary.png"
img_path = os.path.join(root_path ,"yolov10_kidneystone/fine_tune_kidney_stone_data/Bhawarth_in_Hall.png")
result = model.predict(img_path)
print('Finish Prediction on Image')

print('Show Results ')
import matplotlib.pyplot as plt
plt.imshow(result[0].plot())
plt.savefig(img_path.split(".")[0]+'_labelled.png')
plt.show()
print('Done')
