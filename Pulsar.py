import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Data Import
dataset = pd.read_csv('/Users/Prakash/Downloads/pulsar_stars.csv')
dataset.head()  # View first 5 entries in dataset
dataset.shape()  # 17898 rows, 8 variables with one target_class column
dataset = dataset.rename(columns={'target_class': 'target'})  # Simply renaming target column
dataset.isnull().sum()  # Checking null values

# EDA

# Summary of data
plt.figure(figsize=(10, 8))
sns.heatmap(dataset.describe()[1:].transpose(), annot=True, linecolor='w', linewidth=2, cmap=sns.color_palette('Set2'))
plt.title('Summary of data')

# Corrplot to show numeric correlation between variables
corr = dataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
plt.title('Correlation between variables')

# Pair-plot to show relationships between variables
plt.figure(figsize=(20, 20))
sns.pairplot(dataset, hue='target')
plt.show()
