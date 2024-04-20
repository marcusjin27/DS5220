import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
# Convert training history into a DataFrame for easy CSV export
def history_to_dataframe(train_acc, val_acc, train_loss, val_loss):
    # Prepare DataFrame to hold all the data
    df = pd.DataFrame()
    
    for i in range(len(train_acc)):
        df_temp = pd.DataFrame({
            f'Train_Acc_Fold_{i+1}': train_acc[i],
            f'Val_Acc_Fold_{i+1}': val_acc[i],
            f'Train_Loss_Fold_{i+1}': train_loss[i],
            f'Val_Loss_Fold_{i+1}': val_loss[i]
        })
        df = pd.concat([df, df_temp], axis=1)
    
    return df



# Save to CSV


# Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 100), to_categorical(y_test, 100)

# Define CNN model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
acc_per_fold = []
all_train_acc = []
all_val_acc = []
all_train_loss = []
all_val_loss = []
for train_index, val_index in kf.split(x_train):
    train_X, val_X = x_train[train_index], x_train[val_index]
    train_Y, val_Y = y_train[train_index], y_train[val_index]
    
    # Create and train model
    model = create_model()
    print(f'Training fold {fold_no}...')
    history = model.fit(train_X, train_Y, epochs=10, validation_data=(val_X, val_Y), verbose=1)
    
    # Evaluate model
    scores = model.evaluate(val_X, val_Y, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    fold_no += 1
    all_train_acc.append(history.history['accuracy'])
    all_val_acc.append(history.history['val_accuracy'])
    all_train_loss.append(history.history['loss'])
    all_val_loss.append(history.history['val_loss'])
# Assuming 'all_train_acc', 'all_val_acc', 'all_train_loss', 'all_val_loss' are filled as per previous parts
history_df = history_to_dataframe(all_train_acc, all_val_acc, all_train_loss, all_val_loss)
# Average scores
history_df.to_csv('training_history.csv', index=False)
print('Average scores for all folds:')
print(f'Accuracy: {np.mean(acc_per_fold)}% (+- {np.std(acc_per_fold)})')
# Calculate mean accuracy per fold
mean_val_acc = [np.mean(acc) for acc in all_val_acc]
plt.figure(figsize=(8, 6))
sns.boxplot(data=mean_val_acc)


