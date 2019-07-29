import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import scipy
import tensorflow as tf


# min and max value in certain column

class filter:
    def __init__(self, ser):
        self.ser = ser

    def filt_box_minmax(self):
        # min = self.ser.quantile(0.25) * 0.9 #- box_len * 1.5
        min_val = self.ser.mean() * 0.7
        min_val = round(min_val, 2)
        # max = self.ser.quantile(0.75) * 1.1 #+ box_len * 1.5
        max_val = self.ser.mean() * 1.3
        max_val = round(max_val, 2)
        return min_val, max_val

    # Filter for extracting values from certain range
    def filt_range(self):
        low, high = self.filt_box_minmax()
        print(low, high)
        ser_filt = self.ser[(self.ser > low) & (self.ser < high)]
        return ser_filt


def outliers_iqr(ser):
    quartile_1 = ser.quantile(0.25)
    quartile_3 = ser.quantile(0.75)
    iqr = quartile_3 - quartile_1
    lower_bound, upper_bound = (quartile_1 - (iqr * 1.5), quartile_3 + (iqr * 1.5))
    ser_filt = ser[(ser > lower_bound) & (ser < upper_bound)]

    return ser_filt


def com_boxplot(ser1, ser2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.show

def applyModel(df_filt):
    predVal = []

    X = tf.placeholder(tf.float32, shape=[None, 4])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.random_normal([4, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")

    hypothesis = tf.matmul(X, W) + b
    saver = tf.train.Saver()

    for i in range(df_filt.shape[0]):
        model = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(model)

        save_path = "D:/YamP/Python/PyCharm Project/Post Capstone/venv/sources/Study/Big Data/Model/saved.cpkt"
        saver.restore(sess, save_path)

        p1 = df_filt.df_avgTemp_filt.iloc[i]
        p2 = df_filt.df_minTemp_filt.iloc[i]
        p3 = df_filt.df_maxTemp_filt.iloc[i]
        p4 = df_filt.df_rainFall_filt.iloc[i]

        data = ((p1, p2, p3, p4),)
        arr = np.array(data, dtype=np.float32)
        x_data = arr[0:4]

        dict = sess.run(hypothesis, feed_dict={X: x_data})
        predVal.append(dict[0][0])

        # print(p1, p2, p3, p4, dict)

    predVal_ser = pd.Series(predVal)
    sess.close()

    return predVal_ser
