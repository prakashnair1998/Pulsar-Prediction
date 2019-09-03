import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# Data Import
dataset = pd.read_csv('/Users/Prakash/Downloads/pulsar_stars.csv')
dataset.head()  # View first 5 entries in dataset
dataset.shape  # 17898 rows, 8 variables with one target_class column


