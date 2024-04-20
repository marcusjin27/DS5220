import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the training history from CSV
history_df = pd.read_csv('training_history.csv')


# Function to plot the accuracy and loss
def plot_boxplot(history_df):
    val_acc_columns = [col for col in history_df.columns if 'Val_Acc' in col]
    val_acc_data = history_df[val_acc_columns]
    # Setting up the seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create a boxplot for validation accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=val_acc_data)
    plt.title('Box Plot of Validation Accuracy Across Folds')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Folds')
    plt.show()
def plot_accuracy_and_loss(history_df):
    epochs = range(1, len(history_df) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plotting Accuracy
    for col in history_df.columns:
        if 'Acc' in col:
            if 'Train' in col:
                ax1.plot(epochs, history_df[col], label=f'{col}')
            else:
                ax1.plot(epochs, history_df[col], linestyle='--', label=f'{col}')
    ax1.set_title('Model Accuracy per Epoch')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plotting Loss
    for col in history_df.columns:
        if 'Loss' in col:
            if 'Train' in col:
                ax2.plot(epochs, history_df[col], label=f'{col}')
            else:
                ax2.plot(epochs, history_df[col], linestyle='--', label=f'{col}')
    ax2.set_title('Model Loss per Epoch')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Call the function with your DataFrame
plot_accuracy_and_loss(history_df)
plot_boxplot(history_df)
