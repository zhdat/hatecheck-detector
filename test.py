import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    return pd.read_csv(file_path)


def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print(cm)


def test_model(df, text_column, label_column, model, vectorizer):
    df = df.dropna(subset=[text_column])
    X = df[text_column]
    y = df[label_column]

    X = vectorizer.transform(X)
    y_pred = model.predict(X)

    print(classification_report(y, y_pred, zero_division=0))
    plot_confusion_matrix(y, y_pred, labels=model.classes_)


if __name__ == "__main__":
    # Charger le nouveau jeu de données
    df = load_data("data/hatecheck_cases_final_french.csv")

    # Charger le modèle et le vectorizer
    model, vectorizer = load_model_and_vectorizer(
        "out/SVC_model.pkl", "out/vectorizer.pkl"
    )

    # Tester le modèle
    test_model(
        df,
        text_column="test_case",
        label_column="label_gold",
        model=model,
        vectorizer=vectorizer,
    )
