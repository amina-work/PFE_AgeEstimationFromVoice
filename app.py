import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns # visualizing data with stunning default theme
import sklearn # contain algorithms
import warnings
import zipfile #to treat data that's in a zip file.
import os
warnings.filterwarnings('ignore')

# Path to your zip file
zip_file_path = 'path/to/your/archive.zip'

# Directory where you want to extract the files
extract_dir = 'path/to/extract'
# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the specified directory
    zip_ref.extractall(extract_dir)
print("Extraction complete.")