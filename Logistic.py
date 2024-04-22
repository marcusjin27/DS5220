import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import cifar100
# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Flatten the images
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
model.fit(x_train, y_train.ravel())
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
