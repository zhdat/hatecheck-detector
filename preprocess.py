import pandas as pd
import re
import string


def load_data(file_paths):
    dataframes = []
    for file in file_paths:
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding="ISO-8859-1")
            except UnicodeDecodeError:
                print(f"Error: Unable to read {file}")
                continue
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
    else:
        return ""  # or return None


def preprocess_data(df, text_column, label_column):
    df = df.dropna(subset=[text_column, label_column])
    df.loc[:, text_column] = df[text_column].apply(clean_text)
    return df[[text_column, label_column]]


if __name__ == "__main__":
    file_paths = [
        "data-utf8/FTR_new_labels.csv",
        "data-utf8/dataset.csv",
        "data-utf8/hatecheck_cases_final_french.csv",
        "data-utf8/tweets_labeled_4k_cleaned_transformed.csv",
        "data-utf8/french_tweets.csv",
        "data-utf8/french_tweets-003.csv",
        "data-utf8/tweets_cleaned_text.csv",
    ]

    df = load_data(file_paths)
    df = preprocess_data(df, text_column="text", label_column="label")

    df.to_csv("out/preprocessed_data.csv", index=True)
    print("Preprocessing completed and data saved to data/preprocessed_data.csv")
