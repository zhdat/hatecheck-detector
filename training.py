from operator import le
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


def train_and_evaluate_model(
    model, model_name, param_grid, X_train, X_test, y_train, y_test
):
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print(f"Results for {model_name} with best parameters {grid_search.best_params_}:")
    print(classification_report(y_test, y_pred))

    labels = best_model.classes_ if hasattr(best_model, "classes_") else le.classes_
    plot_confusion_matrix(y_test, y_pred, labels=labels)

    joblib.dump(best_model, f"out/{model_name}_model.pkl")
    print(f"{model_name} model saved to out/")


def train_models(df, text_column, label_column):
    df = df.dropna(subset=[text_column])

    X = df[text_column]
    y = df[label_column]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize data
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train).toarray()  # Convert sparse to dense
    X_test = scaler.transform(X_test).toarray()  # Convert sparse to dense

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            },
        ),
        "SVC": (
            SVC(max_iter=10000),
            {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
            },
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(),
            {
                "n_estimators": [50, 100, 200],
                "max_features": ["sqrt", "log2"],
                "max_depth": [None, 10, 20, 30, 40, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "GradientBoostingClassifier": (
            GradientBoostingClassifier(),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 4, 5, 6, 7, 8],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "MultinomialNB": (
            MultinomialNB(),
            {
                "alpha": [0.01, 0.1, 0.5, 1.0],
                "fit_prior": [True, False],
            },
        ),
        "KNeighborsClassifier": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9, 11, 13],
                "weights": ["uniform", "distance"],
                "algorithm": ["brute"],  # Specify brute force algorithm
            },
        ),
        "XGBClassifier": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 7, 8],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
        ),
    }

    for model_name, (model, param_grid) in tqdm(models.items(), desc="Training models"):
        if model_name == "XGBClassifier":
            train_and_evaluate_model(
                model,
                model_name,
                param_grid,
                X_train,
                X_test,
                y_train_encoded,
                y_test_encoded,
            )
        else:
            train_and_evaluate_model(
                model, model_name, param_grid, X_train, X_test, y_train, y_test
            )

    joblib.dump(vectorizer, "out/vectorizer.pkl")
    joblib.dump(scaler, "out/scaler.pkl")
    joblib.dump(le, "out/label_encoder.pkl")
    print("Vectorizer, scaler, and label encoder saved to out/")


if __name__ == "__main__":
    df = load_preprocessed_data("out/tweets_cleaned_text.csv")
    train_models(df, text_column="cleaned_text", label_column="label")
