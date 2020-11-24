import tensorflow as tf
import keras
import keras.backend as k
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
import numpy as np
import math


class Model:

    def __init__(self):
        # Defining data for model
        self.x = np.asarray(tf.random.uniform(minval=0, maxval=1, shape=(6400, 10), dtype=tf.float32))
        self.y = keras.utils.to_categorical(tf.reduce_sum(self.x, axis=-1), num_classes=10)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.x_train = self.x_train.reshape((-1, 10, 1, 1))
        self.x_test = self.x_test.reshape((-1, 10, 1, 1))
        # Hyperparameters
        self.batch_size = 128
        self.epochs = 50
        self.weight_init = RandomNormal()
        self.opt = Adam(lr=0.001)
        # Model
        self.model = self.build_model()
        # Training compiling and printing summary
        self.model.summary()
        self.training_loop()
        self.compile_model()

    def build_model(self):
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(1, 1), activation='relu',
                   kernel_initializer=self.weight_init, input_shape=(10, 1, 1)))
        model.add(Conv2D(64, (1, 1), activation='relu', kernel_initializer=self.weight_init))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Dropout(0.25))
        flatten = Flatten()
        model.add(flatten)
        hidden_layer_1 = Dense(128, activation='relu', kernel_initializer=self.weight_init)
        model.add(hidden_layer_1)
        hidden_layer_2 = Dropout(0.3)
        model.add(hidden_layer_2)
        output_layer = Dense(10, activation='softmax', kernel_initializer=self.weight_init)
        model.add(output_layer)
        return model

    def step(self, real_x, real_y):
        with tf.GradientTape() as tape:
            # Make prediction
            pred_y = self.model(real_x.reshape((-1, 10, 1, 1)))
            # Calculate loss
            model_loss = categorical_crossentropy(real_y, pred_y)

        # Calculate gradients
        model_grads = tape.gradient(model_loss, self.model.trainable_variables)
        # Update model
        self.opt.apply_gradients(zip(model_grads, self.model.trainable_variables))

    def training_loop(self):
        bat_per_epoch = math.floor(len(self.x_train) / self.batch_size)
        for epoch in range(self.epochs):
            print('=', end='')
            for i in range(bat_per_epoch):
                n = i * self.batch_size
                self.step(self.x_train[n:(n + self.batch_size)], self.y_train[n:n + self.batch_size])

    def compile_model(self):
        self.model.compile(loss=Loss_function.loss, optimizer=self.opt, metrics=['accuracy'])

    def return_score(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print('accuracy', score[1])


class Loss_function:

    @staticmethod
    def loss(y_true, y_pred):
        loss = k.sum(k.log(y_true) - k.log(y_pred))
        return loss


class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.categorical_crossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.categorical_accuracy()

    def call(self, step.y_pred, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)


z = Model()
z.return_score()
