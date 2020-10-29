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

tf.compat.v1.disable_eager_execution()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train/255.0, x_test/255.0
# y_train = k.cast(y_train, 'float32')
# y_test = k.cast(y_test, 'float32')
x = tf.random.uniform(minval=0, maxval=1, shape=(1000, 128), dtype=tf.float32)
y = tf.multiply(tf.reduce_sum(x, axis=-1), 5)

weight_init = RandomNormal()
opt = Adam(lr=0.001)
batch_size = 128
epochs = 10
# Build a model
# inputs = Input(shape=(128, ))
# layer1 = Dense(10, activation='relu')(inputs)
# layer2 = Dropout(0.25)(layer1)
# layer3 = Dense(10, activation='relu')(layer2)
# predictions = Dense(10, activation='softmax')(layer3)
# model = Model(inputs=inputs, outputs=predictions)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init, input_shape=(128, 128, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=weight_init))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
input_layer = Input(shape=(128, 128))
model.add(input_layer)
hidden_layer_1 = Dense(128, activation='relu', kernel_initializer=weight_init)
model.add(hidden_layer_1)
hidden_layer_2 = Dropout(0.3)
model.add(hidden_layer_2)
output_layer = Dense(128, activation='softmax', kernel_initializer=weight_init)
model.add(output_layer)


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return k.mean(k.square(y_pred - y_true + k.square(layer)))

    # Return a function
    return loss


def step(x_true, y_true):
    with tf.GradientTape() as tape:
        pred = model(x_true)
        loss = sparse_categorical_crossentropy(y_true, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))


def training_loop():
    bat_per_epoch = math.floor(len(x)/batch_size)
    for epoch in range(epochs):
        print('=', end='')
        for i in range(bat_per_epoch):
            n = i * batch_size
            step(x[n:(n + batch_size)], y[n:n + batch_size])


# Compile the model
model.compile(optimizer=opt,
              loss=custom_loss(input_layer),  # Call the loss function with the selected layer
              metrics=['accuracy'])

# train

model.fit(x, y, steps_per_epoch=100, epochs=10)
# history = model.fit(x, y, epochs=6)
#
# score = model.evaluate(x_test, y_test)
# print('accuracy', score[1])
