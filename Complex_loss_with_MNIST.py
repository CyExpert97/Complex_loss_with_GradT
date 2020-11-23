import keras
from keras.layers import Flatten, Dense, Dropout
import keras.backend as K
from tensorflow.keras.models import Sequential

# Load mnist dataset

mnist = keras.datasets.mnist
# x_train is the list of images y_train is the labels assigned to each image
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalise values to range (0,1)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = K.cast(y_train, 'float32')
y_test = K.cast(y_test, 'float32')

model = Sequential()
# optimizer='adam'
# (28,28) represents the dimensions of image in pixels
input_layer = Flatten(input_shape=(28, 28))
model.add(input_layer)

# Activation function is relu
hidden_layer_1 = Dense(128, activation='relu')
model.add(hidden_layer_1)

# Percentage of nodes destroyed
hidden_layer_2 = Dropout(0.3)
model.add(hidden_layer_2)

# Activation function is softmax
output_layer = Dense(10, activation='softmax')
model.add(output_layer)


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)

    # Return a function
    return loss


# Compile the model
model.compile(optimizer='adam',
              loss=custom_loss(hidden_layer_1),  # Call the loss function with the selected layer
              metrics=['accuracy'])

# Training sets for code with 6 iterations of training
model.fit(x_train, y_train, epochs=6)

# The final test set checking the models performance vs actual test data
score = model.evaluate(x_test, y_test)
print(' accuracy ', score[1])
