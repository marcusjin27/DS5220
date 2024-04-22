import numpy as np
import pandas as pd
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import KFold

# Load and preprocess the CIFAR-100 dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    return x_train, y_train, x_test, y_test

# Define the model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model with 5-fold cross-validation
def train_with_cv(x_train, y_train, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True)
    fold = 0
    histories = []

    for train_index, test_index in kf.split(x_train):
        fold += 1
        print(f"Training fold {fold}/{n_folds}")
        model = create_model(x_train.shape[1], 100)
        history = model.fit(x_train[train_index], y_train[train_index], epochs=10, batch_size=256, 
                            validation_data=(x_train[test_index], y_train[test_index]), verbose=1)
        histories.append(history.history)

    return histories

# Convert training history into a DataFrame for easy CSV export
def history_to_dataframe_and_csv(histories, filename='training_history.csv'):
    df = pd.DataFrame()

    for i, history in enumerate(histories):
        df_temp = pd.DataFrame({
            f'Train_Acc_Fold_{i+1}': history['accuracy'],
            f'Val_Acc_Fold_{i+1}': history['val_accuracy'],
            f'Train_Loss_Fold_{i+1}': history['loss'],
            f'Val_Loss_Fold_{i+1}': history['val_loss']
        })
        df = pd.concat([df, df_temp], axis=1)

    df.to_csv(filename, index=False)
    print(f'Data saved to {filename}')
    return df

# Main function to run the model training and save results
def main():
    x_train, y_train, x_test, y_test = load_data()
    histories = train_with_cv(x_train, y_train)
    df = history_to_dataframe_and_csv(histories)

if __name__ == "__main__":
    main()
