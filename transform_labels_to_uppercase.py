from isort import file
import pandas as pd


def transform_labels_to_uppercase(
    input_file_path, output_file_path, old_label_column, new_label_column
):
    # Load the CSV file
    data = pd.read_csv(input_file_path)

    """ # Transform all labels to uppercase
    data[old_label_column] = data[old_label_column].str.upper() """

    # Rename the column
    data = data.rename(columns={old_label_column: new_label_column})

    # Save the modified dataframe to a new CSV file
    data.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Transformed labels and renamed column saved to {output_file_path}")


if __name__ == "__main__":
    file_path = "data-utf8/tweets_labeled_4k_cleaned_transformed.csv"
    input_file_path = "data-utf8/hatecheck_cases_final_french.csv"  # Change this to your input file path
    output_file_path = "data-utf8/hatecheck_cases_final_french_transformed.csv"  # Change this to your output file path
    old_label_column = "text"  # Change this to your old label column name
    new_label_column = "text"  # Change this to your new label column name

    transform_labels_to_uppercase(
        file_path, file_path, old_label_column, new_label_column
    )
