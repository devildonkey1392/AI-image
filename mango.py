# 一. 載入套件
# 資料處理套件
import cv2
import csv
import random
import time
import numpy as np
import pandas as pd
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import matplotlib.pyplot as plt # plt 用於顯示圖片
import seaborn as sns
# 設定顯示中文字體
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# Keras深度學習模組套件
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import utils as np_utils
from keras import backend as K
from keras import optimizers
# tensorflow深度學習模組套件
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# generated image
train_str = "train.csv"
train_img_dir = "C1-P1_Train/" 
csvfile = open(train_str)
reader = csv.reader(csvfile)
labels_pic = []
labels_level = []
for line in reader:
    labels_pic.append(line[0][:len(line[0])-4])
    labels_level.append(line[1])
csvfile.close() 
labels_pic.pop(0)
labels_level.pop(0)
picnum = len(labels_pic)
print("芒果圖片數量: ",picnum)

X = []
y = []

# 轉換圖片的標籤
for i in range(len(labels_level)):
    labels_level[i] = labels_level[i].replace("A","0")
    labels_level[i] = labels_level[i].replace("B","1")
    labels_level[i] = labels_level[i].replace("C","2")
# 隨機讀取圖片
a = 0
items= []
import random
for a in range(0,picnum):
    items.append(a)

# 製作訓練用資料集及標籤
for i in random.sample(items,picnum):
    img = cv2.imread(train_img_dir + labels_pic[i] + ".jpg")
    res = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    # flip image
    img_flip0=cv2.flip(res,0)
    img_flip1=cv2.flip(res,1)
    img_flip2=cv2.flip(res,-1)
    X.append(img_to_array(img_flip0))
    X.append(img_to_array(img_flip1))
    X.append(img_to_array(img_flip2))
    # rotate image
    (h, w) = res.shape[:2] 
    center = (w // 2, h // 2)
    M_90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    M_270 = cv2.getRotationMatrix2D(center, 270, 1.0)  
    rotated_img_30 = cv2.warpAffine(res, M_90, (w, h))
    rotated_img_270 = cv2.warpAffine(res, M_270, (w, h))
    X.append(img_to_array(rotated_img_30))
    X.append(img_to_array(rotated_img_270))
    res = img_to_array(res)
    X.append(res)    
    for j in range(6):
        y.append(labels_level[i])

y_label_org = y
print(y)
print("x_l: ",len(X))
print("y_l: ", len(y))
# 轉換至array的格式
X = np.array(X)
y = np.array(y)
# 轉換至float的格式
for i in range(len(X)):
    X[i] = X[i].astype('float32')

# 將標籤轉換至float格式
y = tf.strings.to_number(y, out_type=tf.float32)

# 標籤進行one-hotencoding
y = np_utils.to_categorical(y, num_classes = 3)