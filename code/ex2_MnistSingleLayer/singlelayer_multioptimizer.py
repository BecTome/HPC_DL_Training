#!/usr/bin/env python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import read_inputs
import numpy as N
import matplotlib.pyplot as plt
import os

OUTPUT_FOLDER = 'output'
EXERCISE = "ex2_MnistSingleLayer"

N_EPOCHS = 100  
ls_optimizer = [tf.train.GradientDescentOptimizer(0.5),
                 tf.train.MomentumOptimizer(0.05, 0.9),
                 tf.train.AdamOptimizer(0.0005)]

OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, EXERCISE)

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

#read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float32 )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float32 )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0



#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

for optimizer in ls_optimizer:

  with tf.device('/gpu:0'):
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #TRAINING PHASE
    print("TRAINING")

    ls_loss = []
    ls_acc = []
    for epoch in range(N_EPOCHS):
      for i in range(500):
        batch_xs = data[0][0][100*i:100*i+100]
        batch_ys = real_output[100*i:100*i+100]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      #CALCULATING THE LOSS
      loss = sess.run(cross_entropy, feed_dict={x: data[0][0], y_: real_output})
      
      #CHECKING THE ERROR
      # print("ERROR CHECK")

      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      ACC = sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check})
      print(f"EPOCH {epoch}({N_EPOCHS}) -- OPT: {optimizer.__class__.__name__} -- Loss: {loss} --- ACC: {ACC}")

      ls_loss.append(loss)
    ls_acc.append(ACC)
  
  plt.plot(range(N_EPOCHS), ls_loss, label=f"{optimizer.__class__.__name__}")
  # Use log scale to better visualize the loss
  plt.yscale("log")

plt.legend()
plt.title(f"Different Optimizers with their best params")
plt.xlabel("Epochs")
plt.ylabel("Loss")
filename = f"single_layer_multioptimizer".replace(".", "_") + ".pdf"

plt.savefig(os.path.join(OUTPUT_DIR, filename), format="pdf", bbox_inches="tight");

