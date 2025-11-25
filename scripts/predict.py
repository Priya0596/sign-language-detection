import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("./models/model.h5")

def predict_image(image_28x28):
    image = image_28x28.reshape(1, 28, 28, 1)
    pred = model.predict(image)
    return np.argmax(pred)

