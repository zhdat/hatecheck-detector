import streamlit as st
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from lime.lime_text import LimeTextExplainer
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from french_preprocessing.french_preprocessing import FrenchPreprocessing
from unidecode import unidecode


default_stopwords = [
    "y",
    "y'",
    "m",
    "l",
    "d",
    "t",
    "qu",
    "s",
    "c",
    "m'",
    "hein",
    "celle-là",
    "ceux-ci",
    "dring",
    "sa",
    "ollé",
    "en",
    "a",
    "d'",
    "plutôt",
    "auxquels",
    "celles-ci",
    "dès",
    "tel",
    "lui-meme",
    "quelle",
    "les",
    "dont",
    "aie",
    "quand",
    "pour",
    "où",
    "lès",
    "suivant",
    "ho",
    "memes",
    "hem",
    "surtout",
    "mien",
    "tellement",
    "qui",
    "le",
    "quels",
    "tant",
    "une",
    "tien",
    "ohé",
    "i",
    "mêmes",
    "ceux",
    "l'",
    "quelque",
    "si",
    "unes",
    "lequel",
    "tous",
    "chacune",
    "son",
    "que",
    "quel",
    "au",
    "ai",
    "celui-là",
    "chaque",
    "ouste",
    "es",
    "hep",
    "elles-mêmes",
    "lors",
    "cette",
    "cependant",
    "toc",
    "tsouin",
    "chacun",
    "seule",
    "siennes",
    "hum",
    "la",
    "certains",
    "t'",
    "trop",
    "dans",
    "desquels",
    "lui",
    "hors",
    "celles-là",
    "lui-même",
    "pouah",
    "toi-même",
    "boum",
    "vive",
    "rend",
    "mes",
    "vos",
    "nous",
    "qu'",
    "des",
    "tiens",
    "hé",
    "lorsque",
    "zut",
    "vlan",
    "mienne",
    "na",
    "ma",
    "selon",
    "s'",
    "vous-mêmes",
    "eh",
    "ah",
    "ses",
    "meme",
    "lesquels",
    "miens",
    "vôtres",
    "paf",
    "pif",
    "quant-à-soi",
    "tes",
    "c'",
    "sien",
    "ça",
    "lesquelles",
    "tout",
    "telles",
    "même",
    "ces",
    "maint",
    "notre",
    "quanta",
    "elle-même",
    "aupres",
    "bas",
    "votre",
    "plusieurs",
    "moi",
    "par",
    "hurrah",
    "bah",
    "laquelle",
    "auxquelles",
    "vé",
    "peux",
    "pure",
    "tiennes",
    "aujourd'hui",
    "hormis",
    "couic",
    "vous",
    "ore",
    "envers",
    "moindres",
    "aucune",
    "gens",
    "ouias",
    "cela",
    "quelles",
    "aux",
    "pff",
    "etc",
    "toutefois",
    "leurs",
    "ton",
    "clic",
    "las",
    "pfut",
    "t'",
    "toutes",
    "cet",
    "ta",
    "da",
    "toute",
    "aucun",
    "o",
    "sapristi",
    "quoi",
    "desquelles",
    "té",
    "vôtre",
    "euh",
    "pres",
    "as",
    "fi",
    "ci",
    "allo",
    "oh",
    "s'",
    "quiconque",
    "floc",
    "avec",
    "se",
    "bat",
    "tic",
    "jusqu",
    "qu'",
    "unique",
    "certes",
    "celles",
    "dire",
    "tienne",
    "ha",
    "nôtre",
    "jusque",
    "tac",
    "ceux-là",
    "sienne",
    "uns",
    "ouf",
    "moi-même",
    "et",
    "vers",
    "miennes",
    "autrefois",
    "houp",
    "été",
    "à",
    "d'",
    "nouveau",
    "être",
    "peu",
    "dite",
    "s'",
    "dit",
    "tels",
    "ou",
    "toi",
    "entre",
    "avoir",
    "hop",
    "delà",
    "nos",
    "tres",
    "telle",
    "voilà",
    "dessous",
    "soit",
    "autres",
    "psitt",
    "hélas",
    "anterieur",
    "hou",
    "près",
    "auquel",
    "juste",
    "chut",
    "un",
    "stop",
    "eux",
    "ès",
    "vifs",
    "ce",
    "quoique",
    "du",
    "moi-meme",
    "mon",
    "brrr",
    "sous",
    "parmi",
    "deja",
    "déja",
    "celle",
    "siens",
    "suffisant",
    "â",
    "l'",
    "apres",
    "sans",
    "soi-même",
    "là",
    "pur",
    "via",
    "differentes",
    "specifique",
    "holà",
    "tsoin",
    "pan",
    "car",
    "donc",
    "dits",
    "merci",
    "particulièrement",
    "nous-mêmes",
    "personne",
    "allô",
    "soi",
    "voici",
    "sur",
    "vif",
    "celle-ci",
    "malgré",
    "puis",
    "sauf",
    "autre",
    "hui",
    "ceci",
    "leur",
    "celui-ci",
    "necessairement",
    "sacrebleu",
    "hue",
    "eux-mêmes",
    "outre",
    "alors",
    "desormais",
    "plouf",
    "longtemps",
    "malgre",
    "après",
    "de",
    "oust",
    "neanmoins",
    "certain",
    "crac",
    "depuis",
    "olé",
    "hi",
    "te",
    "puisque",
    "m'",
    "me",
    "ô",
    "celui",
    "aussi",
    "rares",
    "chiche",
    "rien",
    "pfft",
    "c'",
    "vu",
    "clac",
    "duquel",
    "aavons",
    "avez",
    "ont",
    "eu",
    "avais",
    "avait",
    "avions",
    "aviez",
    "avaient",
    "eus",
    "eut",
    "eûmes",
    "eûtes",
    "eurent",
    "aurai",
    "auras",
    "aura",
    "aurons",
    "aurez",
    "auront",
    "aurais",
    "aurait",
    "aurions",
    "auriez",
    "auraient",
    "aies",
    "ait",
    "ayons",
    "ayez",
    "aient",
    "eusse",
    "eusses",
    "eût",
    "eussions",
    "eussiez",
    "eussent",
    "ayant",
    "suis",
    "est",
    "sommes",
    "êtes",
    "sont",
    "étais",
    "était",
    "étions",
    "étiez",
    "étaient",
    "fus",
    "fut",
    "fûmes",
    "fûtes",
    "furent",
    "serai",
    "seras",
    "sera",
    "serons",
    "serez",
    "seront",
    "serais",
    "serait",
    "serions",
    "seriez",
    "seraient",
    "sois",
    "soyons",
    "soyez",
    "soient",
    "fusse",
    "fusses",
    "fût",
    "fussions",
    "fussiez",
    "fussent",
    "étant",
]
default_symbols = """#§_-@+=*<>()[]{}/\\"'"""
default_punct = """!;:,.?-..."""

preprocessor = FrenchPreprocessing(
    stopwords=default_stopwords, symbols=default_symbols, punct=default_punct
)


def preprocess_text(text):
    # Remove links
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove mentions (@)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Transliterate accented characters
    text = unidecode(text)

    # Remove special characters
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)

    # Put in lowercase
    text = text.lower()

    return text


sklearn_model_names = [
    "Logistic Regression_optimized",
    "Naive Bayes_optimized",
    "Random Forest_optimized",
    "SVM_optimized",
]
sklearn_models = {
    name: joblib.load(f"./models/{name}.joblib") for name in sklearn_model_names
}

ensemble_model = joblib.load("./models/ensemble_model.joblib")

camembert_tokenizer = CamembertTokenizer.from_pretrained("./models/camembert_model")
camembert_model = CamembertForSequenceClassification.from_pretrained(
    "./models/camembert_model"
)

distilcamembert_tokenizer = CamembertTokenizer.from_pretrained(
    "./models/distilcamenbert-french-hate-speech"
)
distilcamembert_model = CamembertForSequenceClassification.from_pretrained(
    "./models/distilcamenbert-french-hate-speech"
)

dehatebert_tokenizer = AutoTokenizer.from_pretrained("./models/dehatebert-mono-french")
dehatebert_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/dehatebert-mono-french"
)

st.title("Test your classification models")

model_type = st.selectbox(
    "Select the model type", ["Scikit-learn", "Camembert", "Voting Classifier"]
)
model_name = st.selectbox(
    "Select the model",
    sklearn_model_names
    + [
        "camembert_model",
        "Voting Classifier",
        "distilcamenbert_model",
        "dehatebert-mono-french",
    ],
)


input_text = st.text_area("Enter a sentence to test", "")

if st.button("Predict"):
    if input_text:
        preprocessed_text = preprocess_text(input_text)
        preprocessed_text = preprocessor.preprocessing(preprocessed_text)

        progress_bar = st.progress(0)
        progress_bar.progress(10)

        if model_type == "Scikit-learn" and model_name in sklearn_models:
            model = sklearn_models[model_name]
            preprocessed_df = pd.DataFrame({"preprocessed_text": [preprocessed_text]})
            prediction = model.predict(preprocessed_df["preprocessed_text"])
            st.write(f"**{model_name}**: {prediction[0]}")

            explainer = LimeTextExplainer(class_names=["Non-Hateful", "Hateful"])
            vectorizer = model.named_steps["vectorizer"]
            classifier = model.named_steps["classifier"]

            def predict_fn(texts):
                transformed_texts = vectorizer.transform(texts)
                return 1 - classifier.predict_proba(transformed_texts)

            progress_bar.progress(50)

            exp = explainer.explain_instance(
                preprocessed_text, predict_fn, num_features=10
            )
            progress_bar.progress(90)

            st.write("### LIME")
            st.components.v1.html(exp.as_html(), height=500, scrolling=False)
            progress_bar.progress(100)

        elif model_type == "Camembert" and model_name in [
            "camembert_model",
            "distilcamenbert_model",
            "dehatebert-mono-french",
        ]:
            if model_name == "camembert_model":
                tokenizer = camembert_tokenizer
                model = camembert_model
            elif model_name == "distilcamenbert_model":
                tokenizer = distilcamembert_tokenizer
                model = distilcamembert_model
            elif model_name == "dehatebert-mono-french":
                tokenizer = dehatebert_tokenizer
                model = dehatebert_model

            def predict_proba_batch(texts, tokenizer, model, batch_size=16):
                probs_list = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    inputs = tokenizer(
                        batch_texts, return_tensors="pt", padding=True, truncation=True
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = (
                        torch.nn.functional.softmax(outputs.logits, dim=1)
                        .detach()
                        .numpy()
                    )
                    probs_list.extend(probs)
                return np.array(probs_list)

            progress_bar.progress(50)

            prediction_probs = predict_proba_batch(
                [preprocessed_text], tokenizer, model
            )
            prediction = np.argmax(prediction_probs, axis=1)[0]
            st.write(f"**{model_name}**: {['Non-Hateful', 'Hateful'][prediction]}")

            explainer = LimeTextExplainer(class_names=["Non-Hateful", "Hateful"])
            exp = explainer.explain_instance(
                preprocessed_text,
                lambda texts: predict_proba_batch(texts, tokenizer, model),
                num_features=10,
            )
            progress_bar.progress(90)

            st.write("### LIME")
            st.components.v1.html(exp.as_html(), height=500, scrolling=False)
            progress_bar.progress(100)

        elif model_type == "Voting Classifier" and model_name == "Voting Classifier":
            preprocessed_df = pd.DataFrame({"preprocessed_text": [preprocessed_text]})
            prediction = ensemble_model.predict(preprocessed_df["preprocessed_text"])
            st.write(f"**Ensemble Model**: {prediction[0]}")

        else:
            st.write("Model not supported at the moment.")

    else:
        st.write("Please enter a sentence to get a prediction.")
