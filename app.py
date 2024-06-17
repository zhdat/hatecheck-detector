from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models and vectorizer
model_files = [
    "out/GradientBoostingClassifier_model.pkl",
    "out/KNeighborsClassifier_model.pkl",
    "out/LogisticRegression_model.pkl",
    "out/MultinomialNB_model.pkl",
    "out/RandomForestClassifier_model.pkl",
    "out/SVC_model.pkl",
    "out/XGBClassifier_model.pkl",
]
models = {file.split("/")[-1].split("_")[0]: joblib.load(file) for file in model_files}
vectorizer = joblib.load("out/vectorizer.pkl")

# Define weights for each model
model_weights = {
    "GradientBoostingClassifier": 1.0,
    "KNeighborsClassifier": 0.5,
    "LogisticRegression": 1.5,
    "MultinomialNB": 0.5,
    "RandomForestClassifier": 1.0,
    "SVC": 1.5,
    "XGBClassifier": 1.0,
}


# Define a function to convert predictions to numerical values
def prediction_to_numeric(prediction):
    return 1 if prediction == "hateful" else 0


# Define a function to convert numerical values to predictions
def numeric_to_prediction(numeric):
    return "Hateful" if numeric >= 0.5 else "Non-hateful"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])  # type: ignore
def predict():
    if request.method == "POST":
        text = request.form["text"]
        data = [text]
        vect = vectorizer.transform(data).toarray()
        predictions = {}
        weighted_predictions = []

        for model_name, model in models.items():
            prediction = model.predict(vect)[0]
            result = "Hateful" if prediction == "hateful" else "Non-hateful"
            predictions[model_name] = result
            weight = model_weights[model_name]
            weighted_predictions.append(prediction_to_numeric(prediction) * weight)

        # Calculate weighted average prediction
        weighted_average_numeric = np.sum(weighted_predictions) / np.sum(
            list(model_weights.values())
        )
        weighted_average_prediction = numeric_to_prediction(weighted_average_numeric)

        return render_template(
            "index.html",
            input_text=text,
            predictions=predictions,
            average_prediction=weighted_average_prediction,
        )


if __name__ == "__main__":
    app.run(debug=True)
