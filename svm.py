import tensorflow as tf
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np

cifar100 = tf.keras.datasets.cifar100
# Load CIFAR-100 dataset
(X, y), (X_test, y_test) = cifar100.load_data(label_mode='coarse')

# Flatten the images and normalize
X = X.reshape((X.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM with the RBF kernel
clf = svm.SVC(kernel='rbf', gamma='scale')

# Train the SVM
clf.fit(X_train, y_train.ravel())

# Evaluate the model
predicted = clf.predict(X_val)
print(metrics.classification_report(y_val, predicted))