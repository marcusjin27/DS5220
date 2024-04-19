import tensorflow as tf
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

cifar100 = tf.keras.datasets.cifar100
# Load CIFAR-100 dataset
(X, y), (X_test, y_test) = cifar100.load_data(label_mode='coarse')

# Flatten the images and normalize
X = X.reshape((X.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0
print(X)
# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
clf = SGDClassifier(loss='hinge', learning_rate='optimal', max_iter=1, tol=None)
def batch_generator(X, y, batch_size=100):
    n_samples = X.shape[0]
    for offset in range(0, n_samples, batch_size):
        end = offset + batch_size if offset + batch_size < n_samples else n_samples
        yield X[offset:end], y[offset:end].ravel()

# Train incrementally by processing each batch of data

batch_count=0
batch_counts = []
accuracies = []
for X_batch, y_batch in batch_generator(X_train, y_train):
    X_batch
    clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
    print("batch transformed",X_batch)
    if batch_count % 10 == 0:  # Adjust interval as needed
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        batch_counts.append(batch_count)
    
    batch_count += 1
plt.figure(figsize=(10, 6))
plt.plot(batch_counts, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. Batch Count')
plt.xlabel('Batch Count')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()