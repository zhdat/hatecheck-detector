from isort import file
import pandas as pd


def transform_and_group_labels(
    input_file_path, output_file_path, text_column, label_column
):
    # Try reading the CSV file with different encodings
    encodings = ["utf-8", "latin1", "ISO-8859-1"]
    for encoding in encodings:
        try:
            data = pd.read_csv(input_file_path, encoding=encoding)
            break  # If reading is successful, exit the loop
        except UnicodeDecodeError:
            print(
                f"Error reading {input_file_path} with encoding {encoding}. Trying next encoding."
            )
    else:
        print(f"Failed to read {input_file_path} with available encodings.")
        return

    # Replace numeric labels with string labels
    data[label_column] = data[label_column].replace({1: "HATEFUL", 0: "NON-HATEFUL"})

    # Define a function to group labels
    def group_labels(label):
        if label in ["S", "SH", "H", "hateful", "Hateful"]:
            return "hateful"
        elif label in [
            "F",
            "N",
            "non-hateful",
            "Non-hateful",
        ]:
            return "non_hateful"
        elif label in ["ENG", "PUB", "NN", "NAN"]:
            return None
        else:
            return label

    # Apply the function to group labels
    data[label_column] = data[label_column].apply(group_labels)

    # Drop rows with labels that are None
    data = data.dropna(subset=[label_column])

    # Select only the relevant columns
    data = data[[text_column, label_column]]

    # Save the modified dataframe to a new CSV file
    data.to_csv(output_file_path, index=True, encoding="utf-8")
    print(f"Transformed and grouped labels saved to {output_file_path}")


if __name__ == "__main__":
    file_path = "data/hatecheck_cases_final_french.csv"
    input_file_path = "data-utf8/dataset.csv"  # Change this to your input file path
    output_file_path = "data-utf8/dataset.csv"  # Change this to your output file path
    text_column = "test_case"  # Change this to your text column name
    label_column = "label_gold"  # Change this to your label column name

    transform_and_group_labels(file_path, file_path, text_column, label_column)
