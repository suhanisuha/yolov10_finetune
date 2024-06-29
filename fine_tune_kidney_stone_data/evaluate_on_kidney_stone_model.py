from ultralytics import YOLOv10

print('Loading Model ')
checkpoint_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/run/detect/train/xxx.pt"
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')


print('Doing Prediction on a Natural Image')
img_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/kidney_stone_data_roboflow/test/images/1-3-46-670589-33-1-63703718086120120200001-5487554579919763006_png_jpg.rf.9fd67251e99a47dbe83a5db6efe6c016.jpg"
result = model.predict(img_path)
print('Finish Prediction on Image')

print('Show Results ')
import matplotlib.pyplot as plt
plt.imshow(result[0].plot())
plt.show()
print('Done')
