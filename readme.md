# Hate Speech Detection in French on Social Media

## Description

This project aims to detect hate speech in French on social media using various machine learning and deep learning techniques. It was developed as part of an internship at the Norwegian University of Science and Technology (NTNU).

## Fonctionnalités

- Creation and preprocessing of a custom French dataset
- Implementation of machine learning models (SVM, Random Forest, Naive Bayes, Logistic Regression)
- Use of deep learning techniques (LSTM, BiLSTM, GRU)
- Integration of transformer models (DistilCamemBERT, DehateBeRT, CamemBERT)
- Python application to test models on sentences
- Explainable AI with LIME to interpret model decisions

## Project Structure

├── data/
│ ├── cleaned_combined_dataset.csv
│ └── combined_dataset.csv
├── logs/
├── models/
│ ├── camembert_model/
│ │ ├── added_tokens.json
│ │ ├── config.json
│ │ ├── model.safetensors
│ │ ├── sentencepiece.bpe.model
│ │ ├── special_tokens_map.json
│ │ └── tokenizer_config.json
│ ├── dehatebert-mono-french/
│ │ ├── config.json
│ │ ├── model.safetensors
│ │ ├── special_tokens_map.json
│ │ ├── tokenizer.json
│ │ ├── tokenizer_config.json
│ │ └── vocab.txt
│ ├── distilcamenbert-french-hate-speech/
│ │ ├── config.json
│ │ ├── model.safetensors
│ │ ├── sentencepiece.bpe.model
│ │ ├── special_tokens_map.json
│ │ ├── tokenizer.json
│ │ └── tokenizer_config.json
│ ├── BiLSTM_model.keras
│ ├── ensemble_model.joblib
│ ├── GRU_model.keras
│ ├── 'Logistic Regression_optimized.joblib'
│ ├── LSTM_model.keras
│ ├── 'Naive Bayes_optimized.joblib'
│ ├── 'Random Forest_optimized.joblib'
│ └── SVM_optimized.joblib
├── results/
├── static/
├── app_latest.py
├── clean_dataset.ipynb
├── labelling.ipynb
├── paraphrase.ipynb
├── readme.md
├── requirements.txt
├── test.ipynb
├── training-DL.ipynb
├── training-LSTM.ipynb
├── training-ML.ipynb
└── training-transformers.ipynb

## Installation

1. Clone this repository
2. Install dependencies:
   `pip install -r requirements.txt`

## Usage

1. Data preparation: Use `clean_dataset.ipynb` to clean the data.
2. Model training:

- For machine learning models: `training-ML.ipynb`
- For deep learning models: `training-DL.ipynb` and `training-LSTM.ipynb`
- For transformer models: `training-transformers.ipynb`

3. Evaluation and testing: Use `test.ipynb`
4. Application: Run `app_latest.py` to start the Python application: `streamlit run app_latest.py`

## Results

The best performance was achieved with the DistilCamemBERT model:

- Precision: 0.78
- Recall: 0.81
- F1-Score: 0.80

## Contribution

Contributions to this project are welcome. Feel free to open an issue or submit a pull request.

## Authors

- Clément Jantet
- Calliste Ravix

## License

This project is licensed under the MIT License.

## Acknowledgments

We would like to thank the Norwegian University of Science and Technology (NTNU) and ENSICAEN for their support in this project.
