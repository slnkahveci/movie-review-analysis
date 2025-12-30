"""
Task 1: Data Exploration and Insights (15%)
Analyze the IMDB dataset to understand its structure and characteristics.
Suggested analyses:

Class distribution
Average and median text length (overall and per class)
Most frequent words per class
Word clouds per class
Distribution of n-gram frequencies

Please don’t limit yourself to these examples and try to summarize the data with numbers and
graphs. Please highlight some of the most important findings in your report.
Deliverable:

Summarize key findings and observations in your report
Highlight patterns that may influence modeling choices
"""

# TODO: remove most used words that are common stopwords like 'the', 'is', etc. from the n-gram and most frequent words analysis
# TODO: analyse frequently used words that are unique to each class (e.g., words that are frequently used in positive reviews but not in negative reviews, and vice versa)

import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import math

class IMDBDataLoader:
    def __init__(self, file_path, sample_size=50000):
        self.file_path = file_path
        self.sample_size = sample_size

        self.data = None

    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from a given text.

        Args:
            text (str): Input text.
        Returns:
            str: Text with stopwords removed.
        """
        # if not already downloaded, download stopwords
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)


    def load_data(self, remove_stopwords=False) -> pd.DataFrame:
        """
        Load the IMDB dataset from a CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        print(f"Loading data from {self.file_path} with sample size {self.sample_size}...")
        dataframe = pd.read_csv(self.file_path)
        if self.sample_size and self.sample_size < len(dataframe):
            dataframe = dataframe.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        self.data = dataframe
        print(f"Data loaded with shape: {self.data.shape}")

        # remove br tags from reviews
        self.data['review'] = self.data['review'].apply(lambda x: re.sub(r'<br\s*/?>', ' ', x))
        if remove_stopwords:
            print("Removing stopwords from reviews...")
            self.data['review'] = self.data['review'].apply(self._remove_stopwords)

        return self.data

    def get_stats(self) -> dict:
        """
        Get basic statistics of the dataset.

        Returns:
            dict: Dictionary containing basic statistics.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        stats = {
            "num_rows": len(self.data),
            "num_columns": len(self.data.columns),
            "columns": self.data.columns.tolist(),
            "class_distribution": self.data['sentiment'].value_counts().to_dict(),
            "average_text_length":  # chars
            {
                "overall": self.data['review'].apply(len).mean(),
                "per_class": self.data.groupby('sentiment')['review'].apply(lambda x: x.apply(len).mean()).to_dict()
            },
            "average_word_count":  # words
            {
                "overall": self.data['review'].apply(lambda x: len(x.split())).mean(),
                "per_class": self.data.groupby('sentiment')['review'].apply(lambda x: x.apply(lambda y: len(y.split())).mean()).to_dict()
            },
            "median_text_length":
            {
                "overall": self.data['review'].apply(len).median(),
                "per_class": self.data.groupby('sentiment')['review'].apply(lambda x: x.apply(len).median()).to_dict()
            },
            "median_word_count":
            {
                "overall": self.data['review'].apply(lambda x: len(x.split())).median(),
                "per_class": self.data.groupby('sentiment')['review'].apply(lambda x: x.apply(lambda y: len(y.split())).median()).to_dict()     
            }

        }

        def most_frequent_words(texts, n=5):
            all_words = ' '.join(texts).lower()
            words = re.findall(r'\b\w+\b', all_words)
            most_common = Counter(words).most_common(n)
            return dict(most_common)

        stats["most_frequent_words"] = {
            "overall": most_frequent_words(self.data['review']),
            "per_class": self.data.groupby('sentiment')['review'].apply(lambda x: most_frequent_words(x)).to_dict()
        }   

        def ngram_frequencies(texts, n=2):
            all_words = ' '.join(texts).lower()
            words = re.findall(r'\b\w+\b', all_words)
            ngrams = zip(*[words[i:] for i in range(n)])
            ngram_list = [' '.join(ngram) for ngram in ngrams]
            ngram_counts = Counter(ngram_list)
            return dict(ngram_counts.most_common(10))

        stats["ngram_frequencies"] = {
            "overall": {
                "bigrams": ngram_frequencies(self.data['review'], n=2),
                "trigrams": ngram_frequencies(self.data['review'], n=3)
            },
            "per_class": {
                sentiment: {
                    "bigrams": ngram_frequencies(group['review'], n=2),
                    "trigrams": ngram_frequencies(group['review'], n=3)       
                }
                for sentiment, group in self.data.groupby('sentiment')
            }
        }

        def significant_words_per_class(texts, class_labels, n=5):
            """
            Find words that are significantly more frequent in one class vs others.
            Uses log-odds ratio to measure significance.
            """
            class_word_counts = {}
            class_totals = {}

            for label in class_labels.unique():
                class_texts = texts[class_labels == label]
                all_words = ' '.join(class_texts).lower()
                words = re.findall(r'\b\w+\b', all_words)
                word_count = Counter(words)
                class_word_counts[label] = word_count
                class_totals[label] = sum(class_word_counts[label].values())

            significant_words = {}
            labels = list(class_labels.unique())

            for label in labels:
                other_label = [l for l in labels if l != label][0]
                scores = {}

                for word, count in class_word_counts[label].items():
                    # Only consider words appearing at least 10 times
                    if count < 10:
                        continue

                    other_count = class_word_counts[other_label].get(word, 0)

                    # Log-odds ratio with smoothing
                    p1 = (count + 1) / (class_totals[label] + 1)
                    p2 = (other_count + 1) / (class_totals[other_label] + 1)
                    log_odds = math.log(p1 / p2)

                    scores[word] = log_odds

                # Get top N words with highest log-odds (most distinctive for this class)
                top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
                significant_words[label] = {
                    word: round(score, 3) for word, score in top_words
                }

            return significant_words

        stats["significant_words_per_class"] = significant_words_per_class(self.data['review'], self.data['sentiment'])

        def total_unique_words(texts):
            all_words = ' '.join(texts).lower()
            words = set(re.findall(r'\b\w+\b', all_words))
            return len(words)

        stats["total_unique_words"] = {
            "overall": total_unique_words(self.data['review']),
            "per_class": self.data.groupby('sentiment')['review'].apply(lambda x: total_unique_words(x)).to_dict()
        }

        def vocabulary_size(texts, n=1):
            print(f"Calculating {n}-gram vocabulary size...")
            all_words = ' '.join(texts).lower()
            words = re.findall(r'\b\w+\b', all_words)
            ngrams = zip(*[words[i:] for i in range(n)])
            ngram_list = [' '.join(ngram) for ngram in ngrams]
            return len(set(ngram_list))

        stats["vocabulary_size"] = {
            "overall": {
                "1-grams": vocabulary_size(self.data['review'], n=1),
                "2-grams": vocabulary_size(self.data['review'], n=2),
                "3-grams": vocabulary_size(self.data['review'], n=3)
            },
            "per_class": {
                sentiment: {
                    "1-grams": vocabulary_size(group['review'], n=1),
                    "2-grams": vocabulary_size(group['review'], n=2),
                    "3-grams": vocabulary_size(group['review'], n=3)       
                }
                for sentiment, group in self.data.groupby('sentiment')
            }
        }

        return stats

    def visualize_word_clouds(self):
        """
        Visualize word clouds per class.
        """

        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        sentiments = self.data['sentiment'].unique()
        for sentiment in sentiments:
            texts = self.data[self.data['sentiment'] == sentiment]['review']
            all_words = ' '.join(texts).lower()
            # remove any remaining special characters
            all_words = re.sub(r'[^a-zA-Z\s]', '', all_words)

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {sentiment} Reviews')
            plt.show()

    def visualize_ngram_distribution(self, n=2):
        """
        Visualize n-gram frequency distribution.

        Args:
            n (int): The 'n' in n-grams.
        """
        import matplotlib.pyplot as plt

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")


        # TODO: iterate per review, do not combine all reviews into one
        def ngram_frequencies(texts, n=2):
            all_words = ' '.join(texts).lower()
            words = re.findall(r'\b\w+\b', all_words)
            ngrams = zip(*[words[i:] for i in range(n)])
            ngram_list = [' '.join(ngram) for ngram in ngrams]
            ngram_counts = Counter(ngram_list)
            return dict(ngram_counts.most_common(10))
        
        ngram_freq = ngram_frequencies(self.data['review'], n=n)
        ngrams = list(ngram_freq.keys())
        frequencies = list(ngram_freq.values())

        plt.figure(figsize=(12, 6))
        plt.bar(ngrams, frequencies)
        plt.xlabel(f'{n}-grams')
        plt.ylabel('Frequencies')
        plt.title(f'Top {n}-gram Frequencies')
        plt.xticks(rotation=45)
        plt.show()

    def visualize_all(self):
        """
        Visualize all relevant graphs.
        """
        self.visualize_word_clouds()
        self.visualize_ngram_distribution(n=2)
        self.visualize_ngram_distribution(n=3)


if __name__ == "__main__":
    loader = IMDBDataLoader(file_path="dataset/imdb-dataset.csv")
    data = loader.load_data()
    print(data.head())   
    #stats = loader.get_stats()

    import pprint # for better readability
    #pprint.pprint(stats)
    loader.visualize_all()
