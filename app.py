from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import load_model


# At First We Will Load Mnist Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize The Dataset
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One- Hot Encode The Labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Lets See The Shape Of The Dataset
print(f"Shape Of Train Data{x_train.shape}")
print(f"Shape Of Test Data{x_test.shape}")

# Now We Will Build The Model

model = Sequential(
    [
        # In This Model We Will Add Layers
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)
# Now We Will Display Model Architecture
model.summary()

# Complie The Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Lets Start Training The Model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evalutation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save The Model
model.save("mnist_classifier.h5")


loaded_model = load_model("mnist_classifier.h5")

# Now We Will Verify If The Current Model And Loaded Model Have Same Accuracy

loss, accuracy = loaded_model.evaluate(x_test, y_test)
print(f"Loaded Model Accuracy: {accuracy:.4f}")
