import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns # visualizing data with stunning default theme
import sklearn # contain algorithms
import warnings
import zipfile #to treat data that's in a zip file.
import os
warnings.filterwarnings('ignore')


# load dataset
df = pd.read_csv("D:\AMINA\datasets\archive") 
df[df['age'].notna()].head()