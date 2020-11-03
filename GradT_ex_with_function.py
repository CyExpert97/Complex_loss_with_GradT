# install libraries
import tensorflow as tf
import numpy as np
print(tf.__version__)

#Generate sample data
X = tf.constant(value=np.linspace(0, 2, 1000), dtype=tf.float32)
Y = 5*X + 30

# create variables for weight and bias
W = tf.Variable(initial_value=0, trainable=True, name="weight", dtype=tf.float32, )
b = tf.Variable(initial_value=0, trainable=True, name="bias", dtype=tf.float32 )

'''
gredient function to calculate
gradients and loss
'''


def grd(x, y, W, b):
  with tf.GradientTape() as tape:
    tape.watch(W)
    tape.watch(b)
    Y_prd = W*x + b
    loss = tf.reduce_sum(input_tensor=(Y_prd -y)**2) # print(loss)

  grad = tape.gradient(loss, [W, b])
  return grad, loss

STEPS = 100
LEARNING_RATE = .0001

for step in range(STEPS):
#Calculate gradients and loss
  (d_W, d_b), loss = grd(X, Y, W, b)

# update weights
  W = W - d_W * LEARNING_RATE
  b = b - d_b * LEARNING_RATE

# print STEP number and loss
  print("STEP: {} Loss: {}".format(step, loss))

# Print final Loss,weights and bias
print("STEP: {} Loss: {}".format(STEPS,loss))
print("W:{}".format(round(float(W), 4)))
print("b:{}".format(round(float(b), 4)))