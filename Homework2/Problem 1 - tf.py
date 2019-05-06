import time
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
#get mnist data, with one_hot encoding
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#suppress warnings
tf.logging.set_verbosity(old_v)

#learning rate
lr = 0.2
#number of traning epochs
epochs = 20
#number of batch_size
batch_size = 1024
total_batch = int(mnist.train.num_examples/batch_size)
num_steps = epochs * total_batch
print(num_steps)

#network parameters
n_hidden_1 = 100
# n_hidden_2 = 200
num_input = 784
num_classes = 10

tf.reset_default_graph()
#tf graph input
X = tf.placeholder(tf.float32,[None,num_input],name='X')
Y = tf.placeholder(tf.int32,[None,num_classes],name='Y')

#Layers weight & bias
weights = {
    'W1': tf.Variable(tf.random_normal([num_input, n_hidden_1],stddev=0.1),name='W1'),
    # 'W2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=0.1),name='W2'),
    'Wout': tf.Variable(tf.random_normal([n_hidden_1, num_classes],stddev=0.1),name='Wout')
}

biases = {
    'b1': tf.Variable(tf.zeros(shape=[n_hidden_1]),name='b1'),
    # 'b2': tf.Variable(tf.zeros(shape=[n_hidden_2]),name='b2'),
    'bout': tf.Variable(tf.zeros(shape=[num_classes]),name='bout')
}

#define a neural net model
def neural_net(x):
    layer_1_out = tf.nn.relu(tf.add(tf.matmul(x,weights['W1']),biases['b1']))
    # layer_2_out = tf.nn.relu(tf.add(tf.matmul(layer_1_out,weights['W2']),biases['b2']))
    out = tf.add(tf.matmul(layer_1_out,weights['Wout']),biases['bout'])
    return out

#predicted labels
logits = neural_net(X)
Y_hat = tf.nn.softmax(logits)

#define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y),name='cost')
#define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(cost)

#compare the predicted labels with true labels
correct_pred = tf.equal(tf.argmax(Y_hat,1),tf.argmax(Y,1))

#compute the accuracy by taking average
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

#Initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    time_start = time.time()
    for i in range(num_steps):
        # fetch batch
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # run optimization
        _, loss = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
        if i % total_batch == 0:
            time_end = time.time()
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            print("Epoch " + str(i/total_batch+1) +"\tCost: {:.3f}\t".format(loss)+", Accuracy= {:.3f}".format(acc)+
                  "\tTime: {:.3f}\t".format(time_end-time_start))
            time_start = time.time()

    print("Training finished!")

    print("Testing ACcuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))