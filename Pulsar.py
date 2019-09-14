import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
    plt.axhline(dataset[i].mean(), linestyle='dashed')  #

# Violinplot
columns = [x for x in dataset.columns if x not in 'target']
length = len(columns)

for i, j in itertools.zip_longest(columns, range(length)):
    plt.subplot(4, 2, j+1)
    sns.violinplot(x=dataset['target'], y=dataset[i])
    plt.title(i)


# Train-Test Split
X = dataset.drop(columns='target')
Y = dataset['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


# Predictions using KNN and holdout
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
preds = knn.predict(X_test)  # Creates an array of predictions based on our NN model

accuracy_score(preds, Y_test)  # Prints test accuracy (0.970 in this case)


# Predictions using KNN and GridSearchCV
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 15)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X, Y)

knn_gscv.best_params_  # Prints most optimal value of neighbors
knn_gscv.best_score_  # Prints accuracy score with chosen parameters (0.972)

# Prediction using MLP
model = Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(8, 1)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test = np.array(X_train), np.array(X_test)
X_train = np.reshape(X_train, (14318, 8, 1))  # Change dimensions of train and test data to feed in NN
X_test = np.reshape(X_test, (3580, 8, 1))

history = model.fit(X_train, Y_train, epochs=50, batch_size=256, validation_split=0.2)

model.evaluate(X_test, Y_test)  # Evaluate model accuracy (0.977)

# Despite being a simpler model, KNN provides accuracies comparable to an MLP
