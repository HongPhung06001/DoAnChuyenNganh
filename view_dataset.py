import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from work_with_data import view_data

# Xem dữ liệu fer2014
dir_fer = 'DATA/fer2013.csv'
df_fer= pd.read_csv(dir_fer)
# print(df_fer.emotion.unique())
label_fer2013 = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

view_data.view_data(dir_fer, label_fer2013)
view_data.view_image_data(dir_fer, label_fer2013)


# Xem dữ liệu ckextend
dir_ck = 'DATA/ckextended.csv'
df_ck = pd.read_csv(dir_ck)
print(df_ck.emotion.unique())
label_ckextend = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral', 7:'Contempt'}

view_data.view_data(dir_ck, label_ckextend)
view_data.view_image_data(dir_ck, label_ckextend)





