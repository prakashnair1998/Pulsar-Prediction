import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

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
plt.figure(figsize=(10, 10))
sns.pairplot(dataset, hue='target')
plt.show()

# Boxplot
columns = [x for x in dataset.columns if x not in 'target']
length = len(columns)
plt.figure(figsize=(15, 15))

for i, j in itertools.zip_longest(columns, range(length)):
    plt.subplot(4, 2, j+1)
    sns.lvplot(x=dataset['target'], y=dataset[i])
    plt.title(i)
    plt.axhline(dataset[i].mean(), linestyle='dashed')

