import img as img
import tensorflow as tf
import keras
import keras.backend as k
from keras.layers import Dense, Input, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from sklearn.model_selection import train_test_split
import numpy as np
import random
import math

# x and y are defined as our sample data
x = np.asarray(tf.random.uniform(minval=0, maxval=1, shape=(6400, 4), dtype=tf.float32))
y = np.asarray(tf.multiply(tf.reduce_sum(x, axis=-1), 7))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.reshape((-1, 4, 1, 1))
x_test = x_test.reshape((-1, 4, 1, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))


# Hyperparameters
weight_init = RandomNormal()
opt = Adam(lr=0.001)
batch_size = 128
epochs = 50
# [n:(n+batch_size)]

# Builds model that we will use for the training process
model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=weight_init, input_shape=(4, 1, 1)))
model.add(Conv2D(64, (1, 1), activation='relu', kernel_initializer=weight_init))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))
flatten = Flatten()
model.add(flatten)
hidden_layer_1 = Dense(128, activation='relu', kernel_initializer=weight_init)
model.add(hidden_layer_1)
hidden_layer_2 = Dropout(0.3)
model.add(hidden_layer_2)
output_layer = Dense(1, activation='softmax', kernel_initializer=weight_init)
model.add(output_layer)
model.summary()
# keras.utils.plot_model(model, show_shapes=True)


# Define custom loss with added parameter of layer
# def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    # def loss(y_true, y_pred):
    #     return k.mean(k.square(y_pred - y_true + k.square(layer)))

    # Return a function
    # return loss


# Defines function for calculating gradient at each step of learning process
def step(real_x, real_y):
    with tf.GradientTape() as tape:
        # Make prediction
        pred_y = model(real_x.reshape((-1, 4, 1, 1)))
        # Calculate loss
        model_loss = categorical_crossentropy(real_y, pred_y)

    # Calculate gradients
    model_grads = tape.gradient(model_loss, model.trainable_variables)
    # Update model
    opt.apply_gradients(zip(model_grads, model.trainable_variables))


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Training loop
bat_per_epoch = math.floor(len(x_train) / batch_size)
for epoch in range(epochs):
    print('=', end='')
    for i in range(bat_per_epoch):
        n = i*batch_size
        step(x_train[n:n + batch_size], y_train[n:n + batch_size])


# Compile the model
model.compile(optimizer=opt,
              loss=categorical_crossentropy,  # Call the loss function with the selected layer
              metrics=['accuracy'])

print('\n', model.evaluate(x_test, y_test)[1])
