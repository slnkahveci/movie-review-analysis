"""
IMDB Dataset Statistics and Visualization

Computes comprehensive statistics, n-grams, word frequencies, and visualizations
for IMDB sentiment analysis datasets.
"""

import pandas as pd
from collections import Counter
from typing import Optional
from pathlib import Path

import nltk

# Ensure NLTK data is available
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)


class IMDBDataStats:
    """
    Compute statistics and visualizations for IMDB dataset.

    IMPORTANT: This class requires word-based tokenization. Use TextPreprocessor
    with tokenizer_type="word" (default) to generate the required "_words" column.
    Subword tokenizers (BPE, WordPiece, Unigram) are not supported for statistics.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a preprocessed DataFrame.

        Args:
            data: DataFrame from TextPreprocessor.load_data() with tokenizer_type="word"

        Raises:
            ValueError: If data is missing the "_words" column (requires word tokenization)
        """
        if data is None:
            raise ValueError("Data cannot be None.")

        if "_words" not in data.columns:
            raise ValueError(
                "Data must contain a '_words' column. "
                "IMDBDataStats requires word-based tokenization. "
                "Use TextPreprocessor with tokenizer_type='word' (default) before creating an IMDBDataStats instance. "
                "Subword tokenizers (BPE, WordPiece, Unigram) are not supported for statistics."
            )

        self.data = data
        self._word_counters: Optional[dict[str, Counter]] = None
        self.corpus = None  # scattertext corpus (lazy)
        self.stats: Optional[dict] = None

    def _get_word_counters(self) -> dict[str, Counter]:
        """Get word frequency counters per sentiment (cached)."""
        if self._word_counters is not None:
            return self._word_counters

        self._word_counters = {}
        for sentiment in ["positive", "negative"]:
            mask = self.data["sentiment"] == sentiment
            tokens = self.data.loc[mask, "_words"].explode()
            self._word_counters[sentiment] = Counter(tokens.dropna())

        all_tokens = self.data["_words"].explode()
        self._word_counters["overall"] = Counter(all_tokens.dropna())

        return self._word_counters

    def _build_corpus(self):
        """Build scattertext corpus (lazy)."""
        if self.corpus is not None:
            return self.corpus

        import scattertext as st

        print("Building scattertext corpus...")
        self.corpus = st.CorpusFromPandas(
            self.data,
            category_col="sentiment",
            text_col="review",
            nlp=st.whitespace_nlp_with_sentences,
        ).build()

        return self.corpus

    def get_simple_stats(self) -> dict:
        """Compute basic dataset statistics."""
        stats = {}

        # Class distribution
        stats["class_distribution"] = self.data["sentiment"].value_counts().to_dict()

        # Length statistics
        stats["average_text_length"] = {
            "overall": self.data["_char_len"].mean(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_char_len"
            ].mean(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_char_len"
            ].mean(),
        }
        stats["median_text_length"] = {
            "overall": self.data["_char_len"].median(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_char_len"
            ].median(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_char_len"
            ].median(),
        }
        stats["average_word_count"] = {
            "overall": self.data["_word_count"].mean(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_word_count"
            ].mean(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_word_count"
            ].mean(),
        }
        stats["median_word_count"] = {
            "overall": self.data["_word_count"].median(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_word_count"
            ].median(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_word_count"
            ].median(),
        }

        # Vocabulary
        counters = self._get_word_counters()
        stats["vocabulary_size"] = {k: len(v) for k, v in counters.items()}
        stats["most_frequent_words"] = {
            k: v.most_common(10) for k, v in counters.items()
        }
        self.stats = stats
        return stats

    def get_full_stats(self) -> dict:
        """Compute comprehensive dataset statistics."""
        stats = self.get_simple_stats()
        # N-grams
        for n in [2, 3, 4]:
            stats[f"{n}_gram_frequency"] = self._compute_ngram_stats(n)

        # Log-odds ratio
        stats["most_significant_words"] = self._compute_log_odds_ratio()

        self.stats = stats
        return stats

    def _compute_ngram_stats(self, n: int) -> dict[str, list]:
        """Compute n-gram frequencies per sentiment."""
        result = {}

        for sentiment in ["positive", "negative", "overall"]:
            if sentiment == "overall":
                tokens_series = self.data["_words"]
            else:
                mask = self.data["sentiment"] == sentiment
                tokens_series = self.data.loc[mask, "_words"]

            ngram_counter = Counter()
            for tokens in tokens_series:
                ngram_counter.update(nltk.ngrams(tokens, n))

            result[sentiment] = ngram_counter.most_common(10)

        return result

    def _compute_log_odds_ratio(self, top_n: int = 10) -> dict[str, list]:
        """Compute log-odds ratio scores using scattertext."""
        from scattertext import LogOddsRatioUninformativeDirichletPrior

        corpus = self._build_corpus()
        term_freq_df = corpus.get_term_freq_df()

        scorer = LogOddsRatioUninformativeDirichletPrior()
        scores = scorer.get_scores(
            term_freq_df["positive freq"], term_freq_df["negative freq"]
        )

        scores_series = pd.Series(scores, index=term_freq_df.index)

        return {
            "positive": scores_series.nlargest(top_n).index.tolist(),
            "negative": scores_series.nsmallest(top_n).index.tolist(),
        }

    def visualize_word_clouds(self, save_path: Optional[str] = None):
        """Generate word clouds per sentiment."""
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        counters = self._get_word_counters()

        for ax, sentiment in zip(axes, ["positive", "negative"]):
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
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def visualize_scattertext(self, output_html: str = "scattertext.html"):
        """Generate interactive scattertext visualization."""
        import scattertext as st

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

        Path(output_html).write_text(html)
        print(f"Saved to {output_html}")
