import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import copy
from pandas.plotting import scatter_matrix
import scipy
import tensorflow as tf
import seaborn as sns

from sklearn import linear_model
from sklearn import datasets ## imports datasets from scikit-learn

# set path of work directory---------------------------------------
dataAd = 'D:/YamP/Python/PyCharm Project/Post Capstone/venv/sources/Study/Big Data'
base_dir = os.getcwd()
sys.path.append(dataAd)
import KR_Data_Anal as kr

# load data--------------------------------------------------------
data = pd.read_csv(os.path.join(dataAd, 'price data.csv'), sep=',')
df_init = pd.DataFrame(data)
data.head()

# set year column as index and exclude real value------------------
df_init.set_index(df_init['year'], inplace=True)
df = df_init.iloc[:, 1:df_init.shape[1]]
avgPrice = df_init['avgPrice']
df.head()

# filt data and draw boxplot---------------------------------------
df_avgTemp_filt = kr.filter(df['avgTemp'])
df_avgTemp_filt = df_avgTemp_filt.filt_range()

df_minTemp_filt = kr.filter(df['minTemp'])
df_minTemp_filt = df_minTemp_filt.filt_range()

df_maxTemp_filt = kr.filter(df['maxTemp'])
df_maxTemp_filt = df_maxTemp_filt.filt_range()

df_rainFall_filt = kr.filter(df['rainFall'])
df_rainFall_filt = df_rainFall_filt.filt_range()

df_avgPrice_filt = kr.filter(df['avgPrice'])
df_avgPrice_filt = df_avgPrice_filt.filt_range()

# transform series to dataframe-------------------------------------
df_filt_ser_group = [df_avgTemp_filt, df_minTemp_filt, df_maxTemp_filt, df_rainFall_filt, df_avgPrice_filt]

attributes = ["df_avgTemp_filt", "df_minTemp_filt", "df_maxTemp_filt", "df_rainFall_filt", "df_avgPrice_filt"]
df_filt = pd.concat(df_filt_ser_group, axis=1, keys=attributes)  # continue index

# trim NAN data
df_filt = df_filt.dropna(axis=0)

'''
# Draw boxplot------------------------------------------------------
plt.close('all')
attributes_anal = attributes[:-1]
color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange',
         'medians': 'DarkBlue', 'caps': 'Gray'}

df_boxplot      = df.plot.box(color=color, sym='r+')
plt.title("Before filtering")
plt.xlabel("Category")
plt.ylabel("Value")

df_filt_boxplot = df_filt[attributes_anal].plot.box(color=color, sym='r+')
plt.title("After filtering")
plt.xlabel("Category")
plt.ylabel("Value")

# find influence parameters-----------------------------------------
df_filt[attributes_anal].plot.kde()
corr = df_filt.corr(method='pearson')
scatter_matrix(df_filt[attributes_anal], alpha=0.5, figsize=(6,6), diagonal='kde') # digonal='hist', ''kde'
'''

df_filt.pivot_table(index="year", columns="df_avgTemp_filt", values="df_minTemp_filt")


# MLR Method--------------------------------------------------------
df_MLR_X = df_filt.iloc[:, 0:- 1]
df_MLR_y = df_filt["df_avgPrice_filt"]

lm = linear_model.LinearRegression()
model = lm.fit(df_MLR_X, df_MLR_y) # fit as linear model

predictions = lm.predict(df_MLR_X) # find the pred value about df_MLR_X
lm.score(df_MLR_X, df_MLR_y) # the value of R square
lm_coef = lm.coef_
lm_bias = lm.intercept_

predictions_ser = pd.Series(predictions)
avgPrice_ser = pd.Series(df_filt.df_avgPrice_filt)
avgPrice_ser = avgPrice_ser.reset_index(drop=True)

df_MLR = pd.concat([predictions_ser, avgPrice_ser], axis=1)
df_MLR = df_MLR.rename(columns={0: 'pred', 'df_avgPrice_filt': 'real'})
anal_col = ['pred', 'real']

plt.close('all')
scatter_matrix(df_MLR[anal_col], alpha=0.5, figsize=(6,6), diagonal='kde') # digonal='hist', ''kde'

color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange',
         'medians': 'DarkBlue', 'caps': 'Gray'}
df_boxplot      = df_MLR.plot.box(color=color, sym='r+')

sns.lmplot(x='pred', y='real', data=df_MLR) # regression plot
#predVal_ser = kr.applyModel(df_filt)


'''
# save the filtered file as csv
df_filt.to_csv(os.path.join(dataAd, 'price filt.csv'),
               mode='w', #mode='a'
               index=True, header=True,
               na_rep=',') #fill empty space as dash
print('Saved the filtered data')

plt.scatter(predVal_ser, df_filt.avgPrice)
plt.show()
'''