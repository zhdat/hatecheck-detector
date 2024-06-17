import os
import pandas as pd


def convert_to_utf8(input_directory, output_directory):
    # Make sure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the input directory
    files = [f for f in os.listdir(input_directory) if f.endswith(".csv")]

    encodings = ["utf-8", "ISO-8859-1", "latin1"]

    for file in files:
        input_file_path = os.path.join(input_directory, file)
        output_file_path = os.path.join(output_directory, file)

        for encoding in encodings:
            try:
                # Read the CSV file with the current encoding
                data = pd.read_csv(input_file_path, encoding=encoding)
                # Write the CSV file in UTF-8 encoding
                data.to_csv(output_file_path, index=False, encoding="utf-8")
                print(f"Converted {file} to UTF-8 encoding.")
                break  # Exit the loop if successful
            except UnicodeDecodeError:
                print(
                    f"Error reading {input_file_path} with encoding {encoding}. Trying next encoding."
                )
        else:
            print(f"Failed to convert {input_file_path}. Skipping this file.")


if __name__ == "__main__":
    input_directory = "data"  # Change this to your input directory path
    output_directory = "data-utf8"  # Change this to your output directory path
    convert_to_utf8(input_directory, output_directory)
