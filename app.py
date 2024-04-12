######################################## Importing libraries ########################################
import numpy as np # linear algebra
import pandas as pd # data processing for csv files
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns # visualizing data2
import sklearn # contain algorithms
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from scipy.io import wavfile
from tempfile import mktemp
from pydub import AudioSegment
import IPython
import librosa
import IPython.display as ipd
import librosa.display
plt.figure(figsize=(15,4))
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM, Dropout
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


######################################## Data Treatment ########################################
df =  pd.read_csv("D:\AMINA\PFE24\datasets\commonvoice\cv-valid-train.csv")
df_test = pd.read_csv("D:\AMINA\PFE24\datasets\commonvoice\cv-valid-test.csv")

#deleting missing values in relevant columns
print("initial: {} final: {}".format(df.shape, df[df['age'].notna()& df['gender'].notna() & df['accent'].notna()].shape))
print("initial: {} final: {}".format(df_test.shape, df_test[df_test['age'].notna()].shape))
#data cleaning
df = df[['filename','age','gender']]
unbalanced_data = df[df['age'].notna() & df['gender'].notna()]
unbalanced_data.reset_index(inplace=True, drop=True)


######################################## Features Extraction ########################################

######################################## Model Creation ########################################