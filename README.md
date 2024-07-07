## Text Classification with BERT for Arabic Text

This project implements a text classification model using the Bidirectional Encoder Representations from Transformers (BERT) for Arabic text. 
The model is fine-tuned on a custom dataset to classify text into predefined categories.

### Functionality

1. **Model and Tokenizer Initialization:**
    - Loads a pre-trained Arabic BERT model (`asafaya/bert-base-arabic`) from the Transformers library.
    - Initializes a tokenizer associated with the chosen model.
2. **Data Preparation:**
    - Reads training and test data from CSV files.
    - Performs data cleaning, including removing hashtags, URLs, special symbols, and non-Arabic characters.
    - Tokenizes the sentences using the BERT tokenizer.
    - Optionally applies data augmentation techniques to increase training data variety (example: random insertion).
3. **Train-Test Split and Label Encoding:**
    - Splits the training data into training and validation sets using scikit-learn's `train_test_split`.
    - Encodes class labels using scikit-learn's `LabelEncoder`.
4. **Model Training:**
    - Defines a training loop with the following steps for each epoch:
        - Sets the model to training mode.
        - Iterates through batches of training data.
        - Calculates loss for each batch using the model's forward pass.
        - Backpropagates the loss to compute gradients.
        - Updates the model weights using an optimizer (AdamW with weight decay).
    - Performs validation after each epoch to evaluate model performance on unseen data.
5. **Prediction on Test Data:**
    - Prepares the test data similar to the training data (cleaning, tokenization, padding, mask creation).
    - Makes predictions on the test set using the trained model.
    - Converts the predicted labels back to their original class names using the label encoder.
6. **Submission File Generation:**
    - Creates a final submission file in CSV format containing the predicted class labels for the test data.

### Requirements

* Python 
* Libraries:
    * transformers
    * pandas
    * numpy
    * scikit-learn
    * tqdm (optional, for progress bar)

### Usage

1. **Install dependencies:**

```bash
pip install transformers pandas numpy scikit-learn tqdm
```

2. **Replace placeholders:**

- Update file paths in the code for your training (`train.csv`), test (`test.csv`), and sample submission (`sample_submission.csv`) data files.
- Adjust data cleaning steps (`re.sub` patterns) based on your specific data characteristics.

This will generate a submission file named `submission.csv` containing the predicted class labels for your test data.

### License

MIT License
