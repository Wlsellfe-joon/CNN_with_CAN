import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, experimental

img_height = 100
img_width =80
test_batch = 4000
dir = 'C:/Users/YongJun/Desktop/연구/Data/CAN_RGBA_freq/OTIDS_RGBA_freq/DNN_set/Data_Augmentation_Proj_WDOS_WNORM/'

data_name = 'OTIDS_SOUL_After_Data_Aug'
dir = dir + data_name

train = tf.keras.preprocessing.image_dataset_from_directory(dir+'/Train',
                                             shuffle=True,
                                             label_mode='categorical',
                                             validation_split=0.2,
                                             subset="training",
                                             color_mode = "rgba",
                                             seed = 123,
                                             batch_size=32,
                                             image_size=(img_height,img_width))

val = tf.keras.preprocessing.image_dataset_from_directory(dir+'/Train',
                                             shuffle=True,
                                             label_mode='categorical',
                                             validation_split=0.2,
                                             subset="validation",
                                             color_mode = "rgba",
                                             seed = 123,
                                             batch_size=32,
                                             image_size=(img_height,img_width))


test = tf.keras.preprocessing.image_dataset_from_directory(dir+'/Test',
                                            batch_size = test_batch,
                                            color_mode = "rgba",
                                            image_size=(img_height, img_width))
print(train.class_names)

# test dataset 에서 X_true_data와 Y_true_label을 가져옴
for img, lab in test.take(1):
  print("Image shape: ", img.numpy().shape)
  print("Label: ", lab.numpy().shape)

img = img.numpy()
lab = lab.numpy()

#data 확인
print(img.shape)
print(lab.shape)

#Model Setting
METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')]


model = models.Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 4)))

model.add(layers.Conv2D(8,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(16,(3,3), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32,(3,3), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))#total pixels
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2, activation='softmax')) #출력층의 노드 수는 다항분류의 클래스 수와 일치해야 한다.

model.summary()
plot_model(model, to_file='C:/Users/YongJun/Desktop/연구/Data_DNN_모델(현대차프로젝트)/Local/with_Fake_Normal_DoS/'+data_name+'_model.png', show_shapes = True, show_layer_names = True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)
history = model.fit(train, batch_size=32, epochs=10, validation_data = val)


# Accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'Validation'], loc='upper left')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.show()
plt.draw()
fig.savefig('C:/Users/YongJun/Desktop/연구/Data_DNN_모델(현대차프로젝트)/Local/with_Fake_Normal_DoS/'+data_name+'_Acc.png')


#loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.title('LOSS')
plt.show()
plt.draw()
fig.savefig('C:/Users/YongJun/Desktop/연구/Data_DNN_모델(현대차프로젝트)/Local/with_Fake_Normal_DoS/'+data_name+'_Loss.png')


# one hot encoding of Y_true_label for evaluation
lb_test_onehot = to_categorical(lab)

# 테스트셋 정확도 평가_Hyper model
Values = model.evaluate(img,lb_test_onehot)
precision = Values[6] #Precision
recall = Values[7]
F1_Scores = 2*precision*recall/(precision+recall)
print("F1_Score:", F1_Scores)

# Prediction
prediction = model.predict(img)
prediction.shape

#class name 가져오기
class_names = test.class_names
class_names = list(class_names)
class_names

prediction = tf.argmax(prediction, axis=-1)
print(prediction.shape)
confusion_mtx = tf.math.confusion_matrix(lab, prediction)
confusion_mtx


fig = plt.figure()
#plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
plt.draw()
fig.savefig('C:/Users/YongJun/Desktop/연구/Data_DNN_모델(현대차프로젝트)/Local/with_Fake_Normal_DoS/'+data_name+'_Confusion.png')


model.save('C:/Users/YongJun/Desktop/연구/Data_DNN_모델(현대차프로젝트)/Local/with_Fake_Normal_DoS/'+data_name+'_CNN.h5')
