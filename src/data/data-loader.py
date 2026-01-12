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

# TODO do not remove ´` ' when removing punctuation

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
        self.stats = None

        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english")) # words with no or very little meaning
        self._extract_words = nltk.RegexpTokenizer(r"\w+").tokenize # practically word tokenizer, but this will be used in stats only actual tokenization will be subword

    def _remove_stopwords(self, text: str) -> str:
        words = self._extract_words(text)
        return " ".join(w for w in words if w.lower() not in self.stop_words)

    def load_data(self, remove_stopwords=False) -> pd.DataFrame:
        print(
            f"Loading data from {self.file_path} with sample size {self.sample_size}..."
        )
        dataframe = pd.read_csv(self.file_path)
        if self.sample_size and self.sample_size < len(dataframe):
            dataframe = dataframe.sample(n=self.sample_size, random_state=42).reset_index(drop=True)

        dataframe["review"] = dataframe["review"].str.replace(r"<br\s*/?>", " ", regex=True)
        if remove_stopwords:
            print("Removing stopwords from reviews...")
            dataframe["review"] = dataframe["review"].apply(self._remove_stopwords)

        self.data = dataframe

        # save tokens/words into new column , lowercased
        self.data["_words"] = self.data["review"].str.lower().apply(self._extract_words)

        print("Data loaded with shape:", self.data.shape)
        return self.data

    def get_stats(self) -> dict:
        """Get basic statistics of the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        self.stats = {}

        # Precompute lengths to avoid repeated calculations
        self.data["_char_len"] = self.data["review"].str.len()
        self.data["_word_count"] = self.data["_words"].apply(len)

        # should have two lists, one per sentiment
        # dictionary: sentiment -> single flattened list of all words with that sentiment
        all_words_per_sentiment = {}
        for sentiment in ["positive", "negative"]:
            word_lists = self.data[self.data["sentiment"] == sentiment]["_words"].tolist()
            # Flatten the list of lists into a single list
            flat_words = []
            for word_list in word_lists:
                flat_words.extend(word_list)
            all_words_per_sentiment[sentiment] = flat_words

        # Overall: combine all words from both sentiments
        all_words_per_sentiment["overall"] = []
        for word_list in self.data["_words"].tolist():
            all_words_per_sentiment["overall"].extend(word_list)

        self.stats["average_text_length"] = {
            "overall": self.data["_char_len"].mean(),
            "positive": self.data[self.data["sentiment"] == "positive"]["_char_len"].mean(),
            "negative": self.data[self.data["sentiment"] == "negative"]["_char_len"].mean(),    
        }

        self.stats["median_text_length"] = {
            "overall": self.data["_char_len"].median(),
            "positive": self.data[self.data["sentiment"] == "positive"]["_char_len"].median(),
            "negative": self.data[self.data["sentiment"] == "negative"]["_char_len"].median(),
        }

        self.stats["average_word_count"] = {
            "overall": self.data["_word_count"].mean(),
            "positive": self.data[self.data["sentiment"] == "positive"]["_word_count"].mean(),
            "negative": self.data[self.data["sentiment"] == "negative"]["_word_count"].mean(),
        }

        self.stats["median_word_count"] = {
            "overall": self.data["_word_count"].median(),
            "positive": self.data[self.data["sentiment"] == "positive"]["_word_count"].median(),
            "negative": self.data[self.data["sentiment"] == "negative"]["_word_count"].median(),
        }

        def vocabulary_size(sentiment: str) -> int:
            return len(set(all_words_per_sentiment[sentiment]))

        self.stats["vocabulary_size"] = {
            sentiment: vocabulary_size(sentiment) for sentiment in ["overall", "positive", "negative"]
        }

        self.stats["most_frequent_words"] = {
            sentiment: Counter(all_words_per_sentiment[sentiment]).most_common(10) for sentiment in ["overall", "positive", "negative"]
        }

        def ngram_frequency(sentiment: str, n: int) -> list:
            return Counter(nltk.ngrams(all_words_per_sentiment[sentiment], n)).most_common(10)

        self.stats["2_gram_frequency"] = {
            sentiment: ngram_frequency(sentiment, 2) for sentiment in ["overall", "positive", "negative"]
        }

        self.stats["3_gram_frequency"] = {
            sentiment: ngram_frequency(sentiment, 3) for sentiment in ["overall", "positive", "negative"]
        }


        def log_odds_ratio(pos_counts: Counter, neg_counts: Counter, alpha: float = 1.0):
            """Returns dict of word -> log-odds ratio (positive vs negative)."""
            vocab = set(pos_counts.keys()) | set(neg_counts.keys())
            pos_total = sum(pos_counts.values()) + alpha * len(vocab)
            neg_total = sum(neg_counts.values()) + alpha * len(vocab)
            
            scores = {}
            for word in vocab:
                pos_odds = (pos_counts[word] + alpha) / pos_total
                neg_odds = (neg_counts[word] + alpha) / neg_total
                scores[word] = math.log(pos_odds / neg_odds)
            return scores

        # Usage:
        pos_counts = Counter(all_words_per_sentiment["positive"])
        neg_counts = Counter(all_words_per_sentiment["negative"])
        scores = log_odds_ratio(pos_counts, neg_counts)

        # Most positive words
        self.stats["most_positive_words"] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        # Most negative words
        self.stats["most_negative_words"] = sorted(scores.items(), key=lambda x: x[1])[:10]

        

        return self.stats

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
    #loader.visualize_all()
