import pandas as pd


def extract_unique_labels(file_path):
    # Try reading the CSV file with different encodings
    encodings = ["utf-8", "ISO-8859-1"]
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            break  # If reading is successful, exit the loop
        except UnicodeDecodeError:
            print(
                f"Error reading {file_path} with encoding {encoding}. Trying next encoding."
            )
    else:
        print(f"Failed to read {file_path} with available encodings.")
        return

    # Extract all unique labels
    unique_labels = data["label"].unique()

    # Print the unique labels
    print("Unique labels:", unique_labels)


if __name__ == "__main__":
    file_path = "data-utf8/FTR_new_labels.csv"  # Change this to your file path
    extract_unique_labels(file_path)
