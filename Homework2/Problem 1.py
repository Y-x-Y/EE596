import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
IMGS = mnist.train.images
LABS = mnist.train.labels
IMGS_Test = mnist.test.images
LABS_Test = mnist.test.labels

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))

def relu(z):
    return np.maximum(z, 0)

# def relu_prime(z):
#     result = []
#     for t in z:
#         result.append(float(t > 0))
#     return np.array(result, ndmin=2).T

# This one is a lot faster
def relu_prime(z):
    x = np.array(z)
    return np.ceil((abs(x)+x)/2/100)

epochs = 20
lr = 0.2
batch_size = 64
n = mnist.train.num_examples
total_batch = int(n/batch_size)
n_step = epochs * total_batch

beta_1 = 0.9
beta_2 = 0.999
epsilon = 0.00000001

n_input = 784
n_hidden = 100
n_output = 10

w1 = np.random.normal(0, 0.1 ,(n_hidden, n_input))
b1 = np.random.normal(0, 0.1 ,(n_hidden, 1))
w2 = np.random.normal(0, 0.1 ,(n_output, n_hidden))
b2 = np.random.normal(0, 0.1 ,(n_output, 1))

for j in range(epochs):
    time_start = time.time()
    error = 0
    error_t = 0
    cost = 0
    for k in range(0,n,batch_size):
        imgs = IMGS[k:k+batch_size]
        labels = LABS[k:k+batch_size]
        imgs_test = IMGS_Test
        labels_test = LABS_Test
        w1_gd = 0
        w2_gd = 0
        b1_gd = 0
        b2_gd = 0
        t = 0
        a2_m = 0
        a1_m = 0
        a2_v = 0
        a1_v = 0
        a2_m_b = 0
        a1_m_b = 0
        a2_v_b = 0
        a1_v_b = 0

        for x, y in zip(imgs, labels):
            x = np.array(x, ndmin=2).T
            y = np.array(y, ndmin=2).T
            z1 = np.dot(w1, x) + b1
            a1 = relu(z1)
            z2 = np.dot(w2, a1) + b2
            a2 = softmax(z2)

            d2 = a2 - y
            d1 = relu_prime(z1)*np.dot(w2.T, d2)

            w1_gd += np.dot(d1, x.T)
            w2_gd += np.dot(d2, a1.T)
            b1_gd += d1
            b2_gd += d2
            error += int(np.argmax(a2) != np.argmax(y))
            cost = np.sum(y*np.log(a2) + (1-y)*np.log(1-a2))
            # Adam
            t += 1
            a2_m = a2_m * beta_1 + (1-beta_1)*w2_gd
            a1_m = a1_m * beta_1 + (1-beta_1)*w1_gd
            a2_m_b = a2_m_b * beta_1 + (1-beta_1)*b2_gd
            a1_m_b = a1_m_b * beta_1 + (1-beta_1)*b1_gd

            a2_v = a2_v * beta_2 + (1-beta_2)* (w2_gd ** 2)
            a1_v = a1_v * beta_2 + (1-beta_2)* (w1_gd ** 2)
            a2_v_b = a2_v_b * beta_2 + (1-beta_2)* (b2_gd ** 2)
            a1_v_b = a1_v_b * beta_2 + (1-beta_2)* (b1_gd ** 2)

            a2_m_nor = a2_m / (1 - (beta_1 ** t))
            a1_m_nor = a1_m / (1 - (beta_1 ** t))
            a2_v_nor = a2_v / (1 - (beta_2 ** t))
            a1_v_nor = a1_v / (1 - (beta_2 ** t))
            a2_m_b_nor = a2_m_b / (1 - (beta_1 ** t))
            a1_m_b_nor = a1_m_b / (1 - (beta_1 ** t))
            a2_v_b_nor = a2_v_b / (1 - (beta_2 ** t))
            a1_v_b_nor = a1_v_b / (1 - (beta_2 ** t))

        w2 -= (a2_m_nor / (np.sqrt(a2_v_nor) + epsilon)) / batch_size
        w1 -= (a1_m_nor / (np.sqrt(a1_v_nor) + epsilon)) / batch_size
        b2 -= (a2_m_b_nor / (np.sqrt(a2_v_b_nor) + epsilon)) / batch_size
        b1 -= (a1_m_b_nor / (np.sqrt(a1_v_b_nor) + epsilon)) / batch_size

        # w1 -= lr * w1_gd / batch_size
        # w2 -= lr * w2_gd / batch_size
        # b1 -= lr * b1_gd / batch_size
        # b2 -= lr * b2_gd / batch_size
    time_end = time.time()
    print("Epoch: "+str(j+1)+"\tCost: {:.3f}\t".format(-cost)+"\tAccuracy: {:.3f}\t".format(1 - error/n)+
          "\tTime: {:.3f}\t".format(time_end-time_start))

    n_test = mnist.test.num_examples
    for x,y in zip(imgs_test, labels_test):
        x = np.array(x, ndmin=2).T
        y = np.array(y, ndmin=2).T
        z1 = np.dot(w1, x) + b1
        a1 = relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = softmax(z2)
        error_t += int(np.argmax(a2) != np.argmax(y))
    print("Test Accuracy: {:.3f}\t".format(1 - error_t/n_test))
