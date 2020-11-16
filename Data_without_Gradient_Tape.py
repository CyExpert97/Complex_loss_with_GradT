import tensorflow as tf
import keras.backend as k
from keras.layers import Dense, Input
from keras import Model
import numpy as np

tf.compat.v1.disable_eager_execution()
x = tf.random.uniform(minval=0, maxval=1, shape=(128, 128), dtype=tf.float32)
y = tf.multiply(tf.reduce_sum(x, axis=-1), 5)
print(x.shape)
print(type(x))
print(type(y))
print(y.shape)

# Build a model
inputs = Input(shape=(128, ))
layer1 = Dense(64, activation='relu')(inputs)
layer2 = Dense(64, activation='relu')(layer1)
predictions = Dense(64, activation='softmax')(layer2)
model = Model(inputs=inputs, outputs=predictions)


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return k.mean(k.square(y_pred - y_true) + k.square(layer), axis=-1)

    # Return a function
    return loss


# Compile the model
model.compile(optimizer='adam',
              loss=custom_loss(layer1),  # Call the loss function with the selected layer
              metrics=['accuracy'])

# train
model.fit(x, y, steps_per_epoch=500, epochs=10)
