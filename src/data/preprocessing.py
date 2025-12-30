"""
Task 2: Pre-processing (15%)
Design and implement a preprocessing pipeline suitable for text modeling.

Possible components:
Tokenization
Lowercasing
Stopword removal
Stemming or lemmatization


In your report:
Handling punctuation or numbers
Subword tokenization (for Transformer models)
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by applying lowercasing, removing punctuation,
        stopword removal, and lemmatization.

        Args:
            text (str): The input text to preprocess.
        Returns:
            str: The preprocessed text.
        """
        # Lowercasing
        text = text.lower()

        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)

        # Tokenization
        words = text.split()

        # Stopword removal and lemmatization
        processed_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]

        return ' '.join(processed_words)
    
    def preprocess_corpus(self, texts: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess a corpus of texts in a DataFrame.

        Args:
            texts (pd.DataFrame): DataFrame containing the texts to preprocess.
            text_column (str): The name of the column containing the text data.
        Returns:
            pd.DataFrame: DataFrame with an additional column for preprocessed texts.
        """
        # replace the original text column with preprocessed text
        texts[text_column] = texts[text_column].apply(self.preprocess)
        return texts
    

# pretokenize and save to disk as arrow file with huggingface datasets
class PreTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_corpus(self, texts: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Tokenize a corpus of texts in a DataFrame.

        Args:
            texts (pd.DataFrame): DataFrame containing the texts to tokenize.
            text_column (str): The name of the column containing the text data.
        Returns:
            pd.DataFrame: DataFrame with an additional column for tokenized texts.
        """
        texts['tokenized_' + text_column] = texts[text_column].apply(
            lambda x: self.tokenizer.encode(x, truncation=True, padding='max_length')
        )
        return texts
    
    def save_tokenized_corpus(self, texts: pd.DataFrame, file_path: str):
        """
        Save the tokenized corpus to disk as a Hugging Face Dataset arrow file.
        Args:
            texts (pd.DataFrame): DataFrame containing the tokenized texts.
            file_path (str): The file path to save the arrow file.
        """
        from datasets import Dataset
        dataset = Dataset.from_pandas(texts)
        dataset.save_to_disk(file_path)

    def load_tokenized_corpus(self, file_path: str) -> pd.DataFrame:
        """
        Load the tokenized corpus from a Hugging Face Dataset arrow file.
        Args:
            file_path (str): The file path to load the arrow file from.
        Returns:
            pd.DataFrame: DataFrame containing the tokenized texts.
        """
        from datasets import load_from_disk
        dataset = load_from_disk(file_path)
        return dataset.to_pandas()
