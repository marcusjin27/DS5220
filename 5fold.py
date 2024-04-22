import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import KFold
# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the data
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
def create_model():
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(x_train.shape[1],)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(100, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)
scores = []

for train, test in kf.split(x_train):
    model = create_model()
    model.fit(x_train[train], y_train[train], epochs=50, batch_size=256, verbose=1)
    score = model.evaluate(x_train[test], y_train[test], verbose=0)
    print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')
    scores.append(score[1])

print(f'Mean accuracy over {n_folds}-fold cross-validation: {np.mean(scores)}')
final_model = create_model()
final_model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1)
final_score = final_model.evaluate(x_test, y_test, verbose=0)
print(f'Final test loss: {final_score[0]}, Final test accuracy: {final_score[1]}')
