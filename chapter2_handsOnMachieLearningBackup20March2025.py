import pandas as pd
import os
from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the dataset
ML_PATH = '/content/drive/MyDrive/AIandMLLearning/handsOnBookPracticeData/handsonMl2Master/datasets/housing'

# Check if the directory exists (best practice)
if not os.path.exists(ML_PATH):
    print(f"Error: Directory '{ML_PATH}' not found. Please check the path.")
else:
    print(f"Directory '{ML_PATH}' found. Proceeding.")

# Define the function to load the housing data
def load_housing_data(housing_path=ML_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Load the data
housing_data = load_housing_data()

print(housing_data.head())
print(housing_data.info())
print(housing_data["ocean_proximity"].value_counts())
print(housing_data.describe())

housing_data.hist(bins=50, figsize=(20,15))
plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio);
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]





train_set, test_set = split_train_test(housing_data, 0.2)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n");
