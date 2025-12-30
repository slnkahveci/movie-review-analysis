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

import pandas as pd
from collections import Counter
import re
import math
import nltk
from nltk.corpus import stopwords


class IMDBDataLoader:
    def __init__(self, file_path, sample_size=50000):
        self.file_path = file_path
        self.sample_size = sample_size
        self.data = None

        # Download stopwords once at init
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from a given text."""
        words = text.split()
        return " ".join(w for w in words if w.lower() not in self.stop_words)

    def load_data(self, remove_stopwords=False) -> pd.DataFrame:
        """Load the IMDB dataset from a CSV file."""
        print(
            f"Loading data from {self.file_path} with sample size {self.sample_size}..."
        )
        dataframe = pd.read_csv(self.file_path)
        if self.sample_size and self.sample_size < len(dataframe):
            dataframe = dataframe.sample(
                n=self.sample_size, random_state=42
            ).reset_index(drop=True)
        self.data = dataframe
        print(f"Data loaded with shape: {self.data.shape}")

        # remove br tags from reviews
        self.data["review"] = self.data["review"].apply(
            lambda x: re.sub(r"<br\s*/?>", " ", x)
        )
        if remove_stopwords:
            print("Removing stopwords from reviews...")
            self.data["review"] = self.data["review"].apply(self._remove_stopwords)
        return self.data

    def get_stats(self) -> dict:
        """Get basic statistics of the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Precompute lengths to avoid repeated calculations
        self.data["_char_len"] = self.data["review"].apply(len)
        self.data["_word_count"] = self.data["review"].apply(lambda x: len(x.split()))

        def most_frequent_words(texts, n=5):
            words = re.findall(r"\b\w+\b", " ".join(texts).lower())
            return dict(Counter(words).most_common(n))

        # Compute n-grams per review to avoid cross-boundary artifacts
        def ngram_frequencies(texts, n=2):
            ngram_counts = Counter()
            for text in texts:
                words = re.findall(r"\b\w+\b", text.lower())
                ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
                ngram_counts.update(ngrams)
            return dict(ngram_counts.most_common(10))

        def significant_words_per_class(n=5):
            """
            Find words that are significantly more frequent in one class vs others.
            Uses log-odds ratio to measure significance.
            """
            class_word_counts, class_totals = {}, {}
            for label, group in self.data.groupby("sentiment"):
                words = re.findall(r"\b\w+\b", " ".join(group["review"]).lower())
                class_word_counts[label] = Counter(words)
                class_totals[label] = sum(class_word_counts[label].values())

            result = {}
            labels = list(class_word_counts.keys())
            for label in labels:
                other = [l for l in labels if l != label][0]
                scores = {}
                for word, count in class_word_counts[label].items():
                    if count < 10:  # Only consider words appearing at least 10 times
                        continue
                    other_count = class_word_counts[other].get(word, 0)
                    # Log-odds ratio with smoothing
                    p1 = (count + 1) / (class_totals[label] + 1)
                    p2 = (other_count + 1) / (class_totals[other] + 1)
                    scores[word] = math.log(p1 / p2)
                # Get top N words with highest log-odds
                result[label] = {
                    w: round(s, 3)
                    for w, s in sorted(
                        scores.items(), key=lambda x: x[1], reverse=True
                    )[:n]
                }
            return result

        def vocabulary_size(texts, n=1):
            print(f"Calculating {n}-gram vocabulary size...")
            ngram_set = set()
            for text in texts:
                words = re.findall(r"\b\w+\b", text.lower())
                ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
                ngram_set.update(ngrams)
            return len(ngram_set)

        stats = {
            "num_rows": len(self.data),
            "num_columns": len(self.data.columns),
            "columns": [c for c in self.data.columns if not c.startswith("_")],
            "class_distribution": self.data["sentiment"].value_counts().to_dict(),
            "average_text_length": {  # chars
                "overall": self.data["_char_len"].mean(),
                "per_class": self.data.groupby("sentiment")["_char_len"]
                .mean()
                .to_dict(),
            },
            "average_word_count": {  # words
                "overall": self.data["_word_count"].mean(),
                "per_class": self.data.groupby("sentiment")["_word_count"]
                .mean()
                .to_dict(),
            },
            "median_text_length": {
                "overall": self.data["_char_len"].median(),
                "per_class": self.data.groupby("sentiment")["_char_len"]
                .median()
                .to_dict(),
            },
            "median_word_count": {
                "overall": self.data["_word_count"].median(),
                "per_class": self.data.groupby("sentiment")["_word_count"]
                .median()
                .to_dict(),
            },
            "most_frequent_words": {
                "overall": most_frequent_words(self.data["review"]),
                "per_class": self.data.groupby("sentiment")["review"]
                .apply(most_frequent_words)
                .to_dict(),
            },
            "ngram_frequencies": {
                "overall": {
                    "bigrams": ngram_frequencies(self.data["review"], 2),
                    "trigrams": ngram_frequencies(self.data["review"], 3),
                },
                "per_class": {
                    s: {
                        "bigrams": ngram_frequencies(g["review"], 2),
                        "trigrams": ngram_frequencies(g["review"], 3),
                    }
                    for s, g in self.data.groupby("sentiment")
                },
            },
            "significant_words_per_class": significant_words_per_class(),
            "vocabulary_size": {
                "overall": {
                    f"{n}-grams": vocabulary_size(self.data["review"], n)
                    for n in [1, 2, 3]
                },
                "per_class": {
                    s: {
                        f"{n}-grams": vocabulary_size(g["review"], n) for n in [1, 2, 3]
                    }
                    for s, g in self.data.groupby("sentiment")
                },
            },
        }
        return stats

    def visualize_word_clouds(self):
        """Visualize word clouds per class."""
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        for sentiment in self.data["sentiment"].unique():
            texts = self.data[self.data["sentiment"] == sentiment]["review"]
            all_words = re.sub(r"[^a-zA-Z\s]", "", " ".join(texts).lower())
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", stopwords=STOPWORDS
            ).generate(all_words)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud for {sentiment} Reviews")
            plt.show()

    def visualize_ngram_distribution(self, n=2):
        """Visualize n-gram frequency distribution."""
        import matplotlib.pyplot as plt

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Iterate per review to avoid cross-boundary n-grams
        ngram_counts = Counter()
        for text in self.data["review"]:
            words = re.findall(r"\b\w+\b", text.lower())
            ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            ngram_counts.update(ngrams)
        top = dict(ngram_counts.most_common(10))

        plt.figure(figsize=(12, 6))
        plt.bar(top.keys(), top.values())
        plt.xlabel(f"{n}-grams")
        plt.ylabel("Frequencies")
        plt.title(f"Top {n}-gram Frequencies")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def visualize_all(self):
        """Visualize all relevant graphs."""
        self.visualize_word_clouds()
        self.visualize_ngram_distribution(n=2)
        self.visualize_ngram_distribution(n=3)


if __name__ == "__main__":
    loader = IMDBDataLoader(file_path="dataset/imdb-dataset.csv")
    data = loader.load_data(remove_stopwords=True)
    print(data.head())

    stats = loader.get_stats()
    import pprint

    pprint.pprint(stats)
    loader.visualize_all()
