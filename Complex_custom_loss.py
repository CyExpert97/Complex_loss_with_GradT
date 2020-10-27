import tensorflow as tf
import keras
import keras.backend as k
from keras.layers import Dense, Input, Flatten
from keras import Model

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train/255.0, x_test/255.0
# y_train = k.cast(y_train, 'float32')
# y_test = k.cast(y_test, 'float32')
x = tf.random.uniform(minval=0, maxval=1, shape=(1000, 4), dtype=tf.float32)
y = tf.multiply(tf.reduce_sum(x, axis=-1), 5)

# Build a model
inputs = Input(shape=(128, ))
layer1 = Dense(64, activation='relu')(inputs)
layer2 = Dense(64, activation='relu')(layer1)
predictions = Dense(10, activation='softmax')(layer2)
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

history = model.fit(x, y, epochs=6)
#
# score = model.evaluate(x_test, y_test)
# print('accuracy', score[1])
