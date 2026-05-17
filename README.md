# Part 3: NLP and Sequence Modeling

## Project Overview

This project builds an NLP pipeline for customer support sentiment classification. The target column is `sentiment_label`, with three classes: `negative`, `neutral`, and `positive`.

The notebook compares classical text vectorization methods with a sequence-based Bidirectional LSTM model.

## Repository Structure

```text
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── customer_support_text_classification.csv
└── results/
    ├── model_evaluation.csv
    ├── model_evaluation.png
    └── sample_predictions.txt
```

## Setup

In Google Colab, upload `notebook.ipynb` and `customer_support_text_classification.csv`, then run the notebook from top to bottom.

For a local environment:

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Assignment Coverage

### Task 1: Dataset Understanding

The notebook reports the number of records, target classes, sample records, average text length, missing values, duplicate message count, and class distribution.

### Task 2: Text Preprocessing

The preprocessing pipeline lowercases text, removes URLs, removes ticket numbers/digits, removes special characters, tokenizes using whitespace splitting, and removes a small custom stopword list.

### Task 3: Text Vectorization

The notebook creates:

- Bag of Words vectors
- TF-IDF vectors with unigrams and bigrams
- Tokenizer-based padded sequences for the LSTM model

Text must be vectorized because machine-learning models operate on numerical arrays, not raw strings.

### Task 4: Baseline Models

The notebook trains and evaluates:

- Logistic Regression with TF-IDF
- Naive Bayes with Bag of Words
- Dense neural network with TF-IDF

Metrics include accuracy, weighted F1 score, classification report, and confusion matrix.

### Task 5: Sequence Model

The sequence model uses:

```text
Input text -> Tokenizer sequences -> Padding -> Embedding -> Bidirectional LSTM
-> GlobalMaxPooling -> Dense layer -> Softmax output
```

The model uses sparse categorical cross-entropy loss and accuracy as the training metric.

### Task 6: Attention and Transformer Reflection

The notebook explains why RNNs struggle with long-term dependencies, how LSTMs improve memory, what attention solves, and why transformers are important in modern NLP and Generative AI.

## Important Dataset Note

The dataset is synthetic and template-generated. Even after deduplicating exact repeated messages, many examples contain strong sentiment keywords, so very high scores are expected. This is useful for learning the NLP pipeline, but real customer support text would be noisier and harder to classify.
