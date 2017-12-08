# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



import tensorflow as tf



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")



n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100

# We want 10 outputs, because there are 10 possible digits in the MNIST set
n_outputs = 10


reset_graph()


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "BACON3"
logdir = "{}/run-{}/".format(root_logdir, now)

# We know that it will be a matrix with instances along dimension 1 and features along the second
# But we dont know how many instances there are yet, because we havent defined the batch.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")



with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")



im_summary = tf.summary.image('hidden', tf.reshape(hidden1,[-1, 1, n_hidden1, 1]),max_outputs=100)


# Define cost function on which to train the model
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

# Optimiser
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Determine whether the highest logit belongs to the target class
# - boolean of whether or not it is the highest logit (take average for network accuracy)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
acc_summary = tf.summary.scalar('ACC', accuracy)


n_epochs = 20
n_batches = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        acc_str = acc_summary.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        im_str = im_summary.eval(feed_dict={X: X_batch, y: y_batch})
        file_writer.add_summary(acc_str, epoch)
        file_writer.add_summary(im_str, epoch)
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")

