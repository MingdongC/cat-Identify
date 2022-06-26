from PIL import Image
import network_batch
import numpy as np


#在当前文件目录下打开图片
num_px = 64
img = Image.open("datasets/my_image4.jpg").convert("RGB").resize((num_px,num_px))

#reshape图片为适合网络的矩阵维数
my_image = np.array(img).reshape(num_px*num_px*3,-1)
my_image = my_image / 255

#标记图片
my_lable = np.array([[1]])

parameters = network_batch.parameters
my_image_prediction = network_batch.predict(my_image, my_lable, parameters)

result = network_batch.classes[int(np.squeeze(my_image_prediction))].decode("utf-8")
print("img:" ,my_image)
print("predic:" ,my_image_prediction)
print("my prediction for the img is [{}]".format(result))