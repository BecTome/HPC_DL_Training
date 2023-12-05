#!/usr/bin/env python
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import matplotlib.pyplot as plt

OUTPUT_FOLDER = 'output'
EXERCISE = "ex1_GradientDescent"

N_EPOCHS = 1000
ls_lr = [0.05, 0.1, 0.5, 1.0]

OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, EXERCISE)

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

for LEARNING_RATE in ls_lr:
  # Model parameters
  W = tf.Variable([.3], dtype=tf.float32)
  b = tf.Variable([-.3], dtype=tf.float32)
  # Model input and output
  x = tf.placeholder(tf.float32)
  linear_model = W * x + b
  y = tf.placeholder(tf.float32)

  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
  train = optimizer.minimize(loss)

  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong

  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


  ls_loss = []
  for i in range(N_EPOCHS):
    sess.run(train, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    ls_loss.append(curr_loss)

  plt.plot(range(N_EPOCHS), ls_loss, label=f"lr = {LEARNING_RATE}")
  # Use log scale to better visualize the loss
  plt.yscale("log")

plt.legend()
plt.title(f"GradientDescent with different LR")
plt.xlabel("Epochs")
plt.ylabel("Loss")

filename = f"LogGradientDescent_{np.min(ls_lr)}-{np.max(ls_lr)}".replace(".", "_") + ".pdf"

plt.savefig(os.path.join(OUTPUT_DIR, filename), format="pdf", bbox_inches="tight");



