"""
Task 1: Data Exploration and Insights
Efficient IMDB dataset analysis using nltk and scattertext.
"""

import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import scattertext as st
from scattertext import LogOddsRatioUninformativeDirichletPrior


class IMDBDataLoader:
    def __init__(self, file_path: str, sample_size: int = 50000, tokenizer: str = "word"):
        self.file_path = file_path
        self.sample_size = sample_size
        self.data = None
        self.corpus = None
        self.stats = None

        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = RegexpTokenizer(r"\w+") if tokenizer == "word" else None # subword tokenizer can be added later
        self.preprocessor = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize and lowercase text."""
        return [w.lower() for w in self.tokenizer.tokenize(text)]

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Filter out stopwords from token list."""
        return [w for w in tokens if w not in self.stop_words]

    def load_data(self, remove_stopwords: bool = False) -> pd.DataFrame:
        """Load and preprocess the IMDB dataset."""
        print(f"Loading data from {self.file_path}...")

        df = pd.read_csv(self.file_path)

        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)

        # Clean HTML tags
        df["review"] = df["review"].str.replace(r"<br\s*/?>", " ", regex=True)

        # Tokenize
        df["_tokens"] = df["review"].apply(self._tokenize)

        if remove_stopwords:
            print("Removing stopwords...")
            df["_tokens"] = df["_tokens"].apply(self._remove_stopwords)
            df["review"] = df["_tokens"].apply(" ".join)

        # Precompute lengths
        df["_char_len"] = df["review"].str.len()
        df["_word_count"] = df["_tokens"].apply(len)

        self.data = df
        print(f"Loaded {len(df)} reviews.")
        return df

    def _build_corpus(self) -> st.TermDocMatrix:
        """Build scattertext corpus for efficient term analysis."""
        if self.corpus is not None:
            return self.corpus

        print("Building scattertext corpus...")
        self.corpus = st.CorpusFromPandas(
            self.data,
            category_col="sentiment",
            text_col="review",
            nlp=st.whitespace_nlp_with_sentences,
        ).build()

        return self.corpus

    def _get_word_counters(self) -> dict[str, Counter]:
        """Get word frequency counters per sentiment (cached)."""
        if not hasattr(self, "_word_counters"):
            self._word_counters = {}
            for sentiment in ["positive", "negative"]:
                mask = self.data["sentiment"] == sentiment
                tokens = self.data.loc[mask, "_tokens"].explode()
                self._word_counters[sentiment] = Counter(tokens)

            # Overall counter
            all_tokens = self.data["_tokens"].explode()
            self._word_counters["overall"] = Counter(all_tokens)

        return self._word_counters
    
    # Task 1: Data Exploration and Insights
    def get_stats(self) -> dict:
        """Compute comprehensive dataset statistics."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        stats = {}

        # Class distribution
        stats["class_distribution"] = self.data["sentiment"].value_counts().to_dict()

        # Length statistics using vectorized operations
        for metric, func in [("average", "mean"), ("median", "median")]:
            stats[f"{metric}_text_length"] = {
                "overall": getattr(self.data["_char_len"], func)(),
                **self.data.groupby("sentiment")["_char_len"].agg(func).to_dict(),
            }
            stats[f"{metric}_word_count"] = {
                "overall": getattr(self.data["_word_count"], func)(),
                **self.data.groupby("sentiment")["_word_count"].agg(func).to_dict(),
            }

        # Vocabulary statistics
        counters = self._get_word_counters()
        stats["vocabulary_size"] = {
            sentiment: len(counter) for sentiment, counter in counters.items()
        }

        # Most frequent words
        stats["most_frequent_words"] = {
            sentiment: counter.most_common(10)
            for sentiment, counter in counters.items()
        }

        # N-gram frequencies
        for n in [2, 3, 4]:
            stats[f"{n}_gram_frequency"] = self._compute_ngram_stats(n)

        # Log-odds ratio using scattertext
        stats["most_significant_words"] = self._compute_log_odds_ratio()

        self.stats = stats
        return stats

    def _compute_ngram_stats(self, n: int) -> dict[str, list]:
        """Compute n-gram frequencies per sentiment."""
        result = {}

        for sentiment in ["positive", "negative", "overall"]:
            if sentiment == "overall":
                tokens_series = self.data["_tokens"]
            else:
                tokens_series = self.data[self.data["sentiment"] == sentiment][
                    "_tokens"
                ]

            ngram_counter = Counter()
            for tokens in tokens_series:
                ngram_counter.update(nltk.ngrams(tokens, n))

            result[sentiment] = ngram_counter.most_common(10)

        return result

    def _compute_log_odds_ratio(self, top_n: int = 10) -> dict[str, list]:
        """Compute log-odds ratio scores using scattertext."""
        corpus = self._build_corpus()

        # Get term frequencies per category
        term_freq_df = corpus.get_term_freq_df()

        # Use scattertext's log-odds ratio scorer
        scorer = LogOddsRatioUninformativeDirichletPrior()

        # Score terms for positive sentiment
        pos_scores = scorer.get_scores(
            term_freq_df["positive freq"], term_freq_df["negative freq"]
        )

        # Create Series for easy sorting
        scores_series = pd.Series(pos_scores, index=term_freq_df.index)

        return {
            "positive": scores_series.nlargest(top_n).index.tolist(),
            "negative": scores_series.nsmallest(top_n).index.tolist(),
        }

    def visualize_word_clouds(self):
        """Generate word clouds per sentiment."""
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, sentiment in zip(axes, ["positive", "negative"]):
            counters = self._get_word_counters()

            # Filter stopwords for visualization
            word_freq = {
                w: c
                for w, c in counters[sentiment].items()
                if w not in STOPWORDS and len(w) > 2
            }

            wc = WordCloud(
                width=800, height=400, background_color="white", max_words=100
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{sentiment.capitalize()} Reviews")

        plt.tight_layout()
        plt.savefig("wordclouds.png", dpi=150)
        plt.show()

    def visualize_ngram_distribution(self, n: int = 2):
        """Visualize top n-grams."""
        import matplotlib.pyplot as plt

        if self.stats is None:
            self.get_stats()

        ngrams = self.stats[f"{n}_gram_frequency"]["overall"]
        labels = [" ".join(ng[0]) for ng in ngrams]
        values = [ng[1] for ng in ngrams]

        plt.figure(figsize=(12, 6))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel("Frequency")
        plt.title(f"Top {n}-gram Frequencies")
        plt.tight_layout()
        plt.savefig(f"{n}gram_distribution.png", dpi=150)
        plt.show()

    def visualize_scattertext(self, output_html: str = "scattertext.html"):
        """Generate interactive scattertext visualization."""
        corpus = self._build_corpus()

        html = st.produce_scattertext_explorer(
            corpus,
            category="positive",
            category_name="Positive",
            not_category_name="Negative",
            minimum_term_frequency=50,
            width_in_pixels=1000,
            transform=st.Scalers.log_scale_standardize,
        )

        with open(output_html, "w") as f:
            f.write(html)

        print(f"Scattertext visualization saved to {output_html}")

    def visualize_all(self):
        """Generate all visualizations."""
        self.visualize_word_clouds()
        self.visualize_ngram_distribution(n=2)
        self.visualize_ngram_distribution(n=3)
        self.visualize_scattertext()

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


if __name__ == "__main__":
    import pprint

    loader = IMDBDataLoader(file_path="dataset/imdb-dataset.csv")
    data = loader.load_data(remove_stopwords=True)
    print(data.head())

    stats = loader.get_stats()
    pprint.pprint(stats)

    # loader.visualize_all()
