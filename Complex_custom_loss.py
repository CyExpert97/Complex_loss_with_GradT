import img as img
import tensorflow as tf
import keras
import keras.backend as k
from keras.layers import Dense, Input, Flatten, Conv2D, Dropout, MaxPooling2D
from keras import Model
from keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import math
import numpy as np


# x and y are defined as our sample data
x = tf.random.uniform(minval=0, maxval=1, shape=(128, 4, 1), dtype=tf.float32)
y = tf.multiply(tf.reduce_sum(x, axis=-1), 7)


# Hyperparameters
weight_init = RandomNormal()
opt = Adam(lr=0.001)
batch_size = 128
epochs = 10

# Builds model that we will use for the training process
model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=weight_init, input_shape=[128, 4, 1]))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=weight_init))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
flatten = Flatten(k.shape(x)[0])
model.add(flatten)
hidden_layer_1 = Dense(16, input_shape=[4], activation='relu', kernel_initializer=weight_init)
model.add(hidden_layer_1)
hidden_layer_2 = Dropout(0.3)
model.add(hidden_layer_2)
hidden_layer_3 = Dense(32, activation='relu', kernel_initializer=weight_init)
output_layer = Dense(28, activation='softmax', kernel_initializer=weight_init)
model.add(output_layer)


# Define custom loss with added parameter of layer
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return k.mean(k.square(y_pred - y_true + k.square(layer)))

    # Return a function
    return loss


# Defines function for calculating gradient at each step of learning process
def step(x_true, y_true):
    with tf.GradientTape() as tape:
        # Make prediction
        pred_y = model(x_true)
        # Calculate loss
        loss = sparse_categorical_crossentropy(y_true, pred_y)

    # Calculate gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Update model
    opt.apply_gradients(zip(grads, model.trainable_variables))


# Training loop
bat_per_epoch = math.floor(len(x)/batch_size)
for epoch in range(epochs):
    print('=', end='')
    for i in range(bat_per_epoch):
        n = i * batch_size
        step(x[n:n + batch_size], y[n:n + batch_size])


# Compile the model
model.compile(optimizer=opt,
              loss=custom_loss(flatten),  # Call the loss function with the selected layer
              metrics=['accuracy'])

# Fit model
model.fit(x, y, epochs=epochs)
