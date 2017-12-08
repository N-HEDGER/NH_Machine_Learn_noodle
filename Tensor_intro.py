import tensorflow as tf

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



reset_graph()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

sess.close()


# You dont want to repeat sess run all the time, but there is a
# better way
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()



# Instead of running an initializer for each variable, you can instead use
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

# You can also create an interactive shell - the only real difference is that when 
# the session is created it automatically sets itself as the default session - so you 
# No longer have to use 'with'.

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)



# Managing graphs

# Any node you create is automatically added to the default graph

x1=tf.Variable(1)

# In most cases this is fine, but you sometimes want to manage multiple independent graphs.
# You can achieve this by creating a new graph and temporarily making it the default inside a with block.

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3



with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15


# Evaluate y and z without evaluating w and x twice.
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15

from sklearn.datasets import fetch_california_housing

reset_graph()
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]




# Gradient descent regression.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)

reset_graph()

n_epochs = 1000
learning_rate = 0.01

# Constants
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# Variable
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# Matrix multiplication.
y_pred = tf.matmul(X, theta, name="predictions")
# Error
error = y_pred - y
# Compute mean of squared error
mse = tf.reduce_mean(tf.square(error), name="mse")
# Optimiser
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Minimise error.
training_op=optimizer.minimize(mse)

# Initialise globals.
init = tf.global_variables_initializer()


# Run session 
with tf.Session() as sess:
    init.run()
# Loop through
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
            # run the optimiser.
        training_op.run()
    best_theta = theta.eval()









# Placeholder nodes
# Dont perform any computation - they just output what you tell them to at runtime
# By specifying none, we are allowing the dimension to be any size.

reset_graph()


A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1, B_val_2)


# Mini batch gradient descent using tensorboard to visualise
reset_graph()


def fetch_batch(epoch, batch_index, batch_size):
	# Gurantees that the batch will be different.
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

from datetime import datetime

# Get some time information to give the tensorflow saves a unique stamp.
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "BACON3"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

# Placeholder so that values can be added in.
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")



y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

# Evaluate the MSE scalar and write it in a summary file
mse_summary = tf.summary.scalar('MSE', mse)
error_summary = tf.summary.scalar('ERROR', tf.reduce_mean(error))
im_summary = tf.summary.image('IM', tf.reshape(error,[-1, 1, 100, 1]),max_outputs=100)

# Write summaries to log files 
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))



# Run the model
with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown
    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()           

file_writer.close()


best_theta





# 12. Logistic Regression with Mini-Batch Gradient Descent using TensorFlow

from sklearn.datasets import make_moons

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)


plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
plt.legend()
plt.show()


# We need to add a bias feature.
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]

y_moons_column_vector = y_moons.reshape(-1, 1)


# Split into training and test set.
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]


# Define a function for obtaining batches from the data.
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


# Take a small batch
X_batch, y_batch = random_batch(X_train, y_train, 5)
X_batch


# Define model
reset_graph()
n_inputs = 2
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
logits = tf.matmul(X, theta, name="logits")
y_proba = tf.sigmoid(logits)


loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default


learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()



n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})




