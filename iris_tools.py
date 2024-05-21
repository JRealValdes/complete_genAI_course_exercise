import tensorflow as tf
import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = "iris_model.h5"

def create_model():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=100)
    return model

def inference(X_data):
    if not os.path.exists(MODEL_PATH):
        model = create_model()
        model.save(MODEL_PATH)
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(np.array([X_data]))
    predicted_class = np.argmax(prediction)
    return int(predicted_class) 
