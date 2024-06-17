import pandas as pd
import matplotlib.pyplot as plt

import chardet

DEBUG = False

file = "2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv"

if DEBUG:
    with open(file, "rb") as f:
        result = chardet.detect(f.read())
        encoding = result["encoding"]

    try:
        df = pd.read_csv(file, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="latin-1")

df = pd.read_csv(file)
print("read ok !")

label_counts = df["label"].value_counts()

label_counts.plot(kind="bar", rot=0)
plt.xlabel("Labels")
plt.ylabel("Counts")
plt.show()
