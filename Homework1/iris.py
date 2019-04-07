import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import get_dummies
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file_name = "iris.data.csv"

#load the dataset into memory
dataset = pd.read_csv(file_name,header=None,\
                      names=['sepal_length','sepal_width','petal_length','petal_width','species'])
#check several examples of data
dataset.head()

#simple function to process the data.
"""
input:
    data: iris dataset
    num_features: 2 => select only petal width and petal width.
                  4 => select all features
    flower: 'Iris-setosa' => labels of Iris-setosa will be 1 and others will be 0
            'Iris-virginica' => labels of Iris-virginica will be 1 and others will be 0
            'None' => use one-hot encoding to represent the labels

return:
    x: normalized & shuffled data
    y: labels
"""
def data_preprocessing(data,num_features,flower=None):
    if num_features==2:
        features = data.columns[2:4]
    else:
        features = data.columns[0:4]
    labels = dataset.columns[4]
    print(features)
    print(labels)
    #normalize the data
    data_norm = pd.DataFrame(data)
    for feature in features:
        data[feature] = (data[feature]-data[feature].mean())/data[feature].std()

    #shuffle the data
    indices = data_norm.index.tolist()
    indices = np.array(indices)
    np.random.shuffle(indices)
    x = data_norm.reindex(indices)[features]
    y = data_norm.reindex(indices)[labels]
    if flower=='Iris-setosa':
        for i in range(len(y)):
            y[i]=1 if y[i]=='Iris-setosa' else 0
        y = y.values.reshape(len(y),1)
    elif flower=='Iris-virginica':
        for i in range(len(y)):
            y[i]=1 if y[i]=='Iris-virginica' else 0
        y = y.values.reshape(len(y),1)
    else:
        y = get_dummies(y)
    return x,y

# Part a
x_1,y_1 = data_preprocessing(dataset,2,flower='Iris-setosa')
#generate traning and validation sets
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1,y_1,test_size=0.3)

train_size = len(x_train_1)
epoch = 100
NUM_ITER = train_size * epoch
lr = 0.01
acc = np.zeros(epoch)

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 2], name='feature')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='label')
# single perceptron
W = tf.Variable(tf.random_normal([2,1]),tf.float32,name='W')
b = tf.Variable(tf.zeros([1]),tf.float32,name='b')

out = tf.matmul(X,W)+b
Y_hat = tf.nn.sigmoid(out)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=out))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.round(Y_hat),tf.round(Y))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(NUM_ITER):
        sess.run(train_step, feed_dict={X: x_train_1, Y: y_train_1})
        if i%train_size == 0:
            acc[i//train_size] = sess.run(accuracy, feed_dict={X: x_train_1, Y: y_train_1})
            print("epoch"+str(i/train_size+1)+"\tAccuracy={:.3f}".format(acc[i//train_size]))
    w_ = np.squeeze(sess.run(W))
    b_ = np.squeeze(sess.run(b))
    outcome = sess.run(accuracy, feed_dict={X: x_test_1, Y: y_test_1})

print("Test\t"+"\tAccuracy={:.3f}".format(outcome))

n = []
for i in range(100):
    n.append(i+1)
plt.plot(n, acc, 'r-o')
plt.show()
plot_x1 = np.array([min(x_1['petal_length'])-0.3, max(x_1['petal_length'])+0.3])
plot_y1 = 1 / w_[1] * (-w_[0] * plot_x1 - b_)
plt.plot(plot_x1, plot_y1, color='k', linewidth=2)
plt.scatter(x_1['petal_length'], x_1['petal_width'], c=np.squeeze(y_1), s=20, cmap='viridis')
plt.xlim([min(x_1['petal_length'])-0.3, max(x_1['petal_length'])+0.3]);
plt.ylim([min(x_1['petal_width'])-0.3, max(x_1['petal_width'])+0.3]);
plt.show()

# Part b
x_2,y_2 = data_preprocessing(dataset,2,flower='Iris-virginica')
#generate traning and validation sets
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2,y_2,test_size=0.3)

train_size = len(x_train_2)
epoch = 100
NUM_ITER = train_size * epoch
lr = 0.01
acc = np.zeros(epoch)

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 2], name='feature')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='label')
# single perceptron
W = tf.Variable(tf.random_normal([2,1]),tf.float32,name='W')
b = tf.Variable(tf.zeros([1]),tf.float32,name='b')

out = tf.matmul(X,W)+b
Y_hat = tf.nn.sigmoid(out)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=out))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.round(Y_hat),tf.round(Y))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(NUM_ITER):
        sess.run(train_step, feed_dict={X: x_train_2, Y: y_train_2})
        if i%train_size == 0:
            acc[i//train_size] = sess.run(accuracy, feed_dict={X: x_train_2, Y: y_train_2})
            print("epoch"+str(i/train_size+1)+"\tAccuracy={:.3f}".format(acc[i//train_size]))
    w_ = np.squeeze(sess.run(W))
    b_ = np.squeeze(sess.run(b))
    outcome = sess.run(accuracy, feed_dict={X: x_test_2, Y: y_test_2})

print("Test\t"+"\tAccuracy={:.3f}".format(outcome))

n = []
for i in range(100):
    n.append(i+1)
plt.plot(n, acc, 'r-o')
plt.show()
plot_x1 = np.array([min(x_2['petal_length'])-0.3, max(x_2['petal_length'])+0.3])
plot_y1 = 1 / w_[1] * (-w_[0] * plot_x1 - b_)
plt.plot(plot_x1, plot_y1, color='k', linewidth=2)
plt.scatter(x_2['petal_length'], x_2['petal_width'], c=np.squeeze(y_2), s=20, cmap='viridis')
plt.xlim([min(x_2['petal_length'])-0.3, max(x_2['petal_length'])+0.3]);
plt.ylim([min(x_2['petal_width'])-0.3, max(x_2['petal_width'])+0.3]);
plt.show()

# Part c
x_3,y_3 = data_preprocessing(dataset,4)
#generate traning and validation sets
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_3,y_3,test_size=0.3)

train_size = len(x_train_3)
epoch = 100
NUM_ITER = train_size * epoch
lr = 0.01
acc = np.zeros(epoch)
num_class = y_3.shape[1]

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 4], name='feature')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='label')
W1 = tf.Variable(tf.random_normal([4, 256]), name='weights1')
W2 = tf.Variable(tf.random_normal([256, 128]), name='weights2')
W3 = tf.Variable(tf.random_normal([128, 3]), name='weights3')
b1 = tf.Variable(tf.random_normal([256]), name='bias1')
b2 = tf.Variable(tf.random_normal([128]), name='bias2')
b3 = tf.Variable(tf.random_normal([3]), name='bias3')

layer_1 = tf.matmul(X,W1)+b1
layer_1 = tf.nn.relu(layer_1)
layer_2 = tf.matmul(layer_1,W2)+b2
layer_2 = tf.nn.relu(layer_2)
out = tf.matmul(layer_2,W3)+b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=out))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(out, 1),tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(NUM_ITER):
        sess.run(train_step, feed_dict={X: x_train_3, Y: y_train_3})
        if i%train_size == 0:
            acc[i//train_size] = sess.run(accuracy, feed_dict={X: x_train_3, Y: y_train_3})
            print("epoch"+str(i/train_size+1)+"\tAccuracy={:.3f}".format(acc[i//train_size]))
    outcome = sess.run(accuracy, feed_dict={X: x_test_3, Y: y_test_3})

print("Test\t"+"\tAccuracy={:.3f}".format(outcome))

n = []
for i in range(100):
    n.append(i+1)
plt.plot(n, acc, 'r-o')
plt.show()