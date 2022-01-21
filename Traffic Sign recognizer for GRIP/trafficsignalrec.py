# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Conv2D, MaxPool2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32



# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []
classes = 43

for i in range(classes):
    path = os.path.join(DIRECTORY,'train',str(i))
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(30, 30))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(i)


data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=0)
# perform one-hot encoding on the labels

trainY = to_categorical(trainY, 43)
testY = to_categorical(testY, 43)

print('Network .......')
model = Sequential()
print('1st conv')
model.add(Conv2D(32,(5,5),activation= 'relu',input_shape=trainX.shape[1:]))
model.add(Conv2D(32,(5,5),activation= 'relu'))

model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
print('2nd conv')
model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(Conv2D(64,(3,3),activation= 'relu'))

model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

print('output layer')
model.add(Dense(43,activation='softmax'))


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
H = model.fit(
	trainX,
	trainY,
	batch_size=BS,
	validation_data=(testX, testY),
	epochs=EPOCHS )



# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# serialize the model to disk
print("[INFO] saving trafic sign recognizer model...")
model.save("trafic sign recognizer.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")