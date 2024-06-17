import joblib
import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection, preprocessing
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# List of model files
model_files = [
    "out/GradientBoostingClassifier_model.pkl",
    "out/KNeighborsClassifier_model.pkl",
    "out/LogisticRegression_model.pkl",
    "out/MultinomialNB_model.pkl",
    "out/RandomForestClassifier_model.pkl",
    "out/SVC_model.pkl",
    "out/XGBClassifier_model.pkl",
]

# Dummy dataset for demonstration purposes
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


def create_keras_model(input_shape, num_classes):
    model = Sequential(
        [
            keras.Input(shape=input_shape),
            Dense(10, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def convert_and_save_model(model_path, output_dir, model_name):
    # Load the sklearn model
    sklearn_model = joblib.load(model_path)

    # Create a dummy Keras model
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y))
    keras_model = create_keras_model(input_shape, num_classes)

    # Train the Keras model on the dummy dataset
    keras_model.fit(X_train, y_train, epochs=10)

    # Save the Keras model
    keras_model_path = os.path.join(output_dir, f"{model_name}.keras")
    keras_model.save(keras_model_path)

    # Convert the Keras model to TensorFlow.js format
    tfjs_output_dir = os.path.join(output_dir, f"{model_name}_tfjs")
    os.makedirs(tfjs_output_dir, exist_ok=True)
    os.system(
        f"tensorflowjs_converter --input_format=keras --output_format=tfjs_graph_model {keras_model_path} {tfjs_output_dir}"
    )


# Ensure the tensorflow directory exists
os.makedirs("tensorflow", exist_ok=True)

for model_file in model_files:
    model_name = model_file.split("/")[-1].split("_")[0]
    output_dir = "tensorflow"
    convert_and_save_model(model_file, output_dir, model_name)
