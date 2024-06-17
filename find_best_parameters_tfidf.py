from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from stop_words import get_stop_words

# Exemple de données
data = pd.read_csv("out/tweets_cleaned_text.csv")

# Supprimer les lignes avec des valeurs manquantes dans la colonne de texte
data = data.dropna(subset=["cleaned_text"])

# Assurer que la colonne de texte est de type chaîne
data["cleaned_text"] = data["cleaned_text"].astype(str)

# Obtenir les mots vides en français
french_stop_words = get_stop_words("french")


# Définir un pipeline avec TfidfVectorizer et LogisticRegression
pipeline = Pipeline(
    [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
)

# Définir les paramètres de la grille de recherche
parameters = {
    "tfidf__max_features": [1000, 5000, 10000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.5, 0.8, 1.0],
    "tfidf__stop_words": [None, french_stop_words],
    "tfidf__sublinear_tf": [True, False],
}

# Effectuer la recherche en grille
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(data["cleaned_text"], data["label"])

# Afficher les meilleurs paramètres trouvés
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
