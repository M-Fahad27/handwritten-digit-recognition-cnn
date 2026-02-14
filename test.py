from tensorflow.keras.models import load_model
import numpy as np
import cv2

# At First We Will Load The Model
loaded_model = load_model("mnist_classifier.h5")

# Now We Will Load Our Image Containing Digit
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28
img = cv2.resize(img, (28, 28))

# 3. Invert colors, Picture Must be Black Background With White Writing
img = 255 - img

# Normalize
img = img.astype("float32") / 255.0

# Reshape To Match Model Input
img = img.reshape(1, 28, 28, 1) 

# Lets Test And Make Prediction
prediction = loaded_model.predict(img)
digit = np.argmax(prediction)

print(f"Predicted Digit: {digit:.2f}")
