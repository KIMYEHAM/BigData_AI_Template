import tensorflow as tf
import numpy as np
import pandas as pd
import os

'''
# 1. Develope Tensorflow Model
model = tf.global_variables_initializer()

# set pwd
dataAd = 'D:/YamP/Python/PyCharm Project/Post Capstone/venv/sources/Study/Big Data'
base_dir = os.getcwd()

data = pd.read_csv(os.path.join(dataAd, 'price data.csv'), sep=',')
data.head()

# use Total Data with np
xy = np.array(data, dtype=np.float32)

# seperate prameters and predict value
x_data = xy[:, 1:-1]
y_data = xy[:, [-1]]

# set place holder
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# set hypothesis function
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(cost)

# construct tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print("#", step, "손실 비용: ", cost_)
        print("- 배추가격:",hypo_[0])

saver = tf.train.Saver()
save_path = saver.save(sess, "D:/YamP/Python/PyCharm Project/Post Capstone/venv/sources/Study/Big Data/saved.cpkt")
print("학습된 모델을 저장했습니다.")
'''

# 2. Load Tensorflow Model
# set placeholder same as saved model
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# input from user
avg_temp  = float(input('평균 온도: '))
min_temp  = float(input('최저 온도: '))
max_temp  = float(input('최고 온도: '))
rain_fall = float(input('강수량: '))

with tf.Session() as sess:
    sess.run(model)

    save_path = "D:/YamP/Python/PyCharm Project/Post Capstone/venv/sources/Study/Big Data/saved.cpkt"
    saver.restore(sess, save_path)

    data = ((avg_temp, min_temp, max_temp, rain_fall), )
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print(dict[0])