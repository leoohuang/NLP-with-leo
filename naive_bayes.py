"""Lecture 01 demo: Multinomial Naive Bayes for tiny text classification.

Run the built-in demo:
    python3 naive_bayes.py

Run with TSV files:
    python3 naive_bayes.py --train examples/movie_reviews.tsv --test examples/movie_reviews.tsv

TSV format:
    label<TAB>text
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


LabeledText = tuple[str, str]


def tokenize(text: str) -> list[str]:
    """A tiny tokenizer for teaching demos.

    If a Chinese sentence already contains spaces, we keep space-separated words.
    Otherwise, contiguous English/digit chunks are kept and Chinese characters are
    split one by one. Real NLP projects should use a stronger tokenizer.
    """

    text = text.strip().lower()
    if not text:
        return []
    if " " in text:
        return [token for token in re.split(r"\s+", text) if token]
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


def load_tsv(path: Path) -> list[LabeledText]:
    corpus: list[LabeledText] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        try:
            label, text = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} should be: label<TAB>text") from exc
        corpus.append((text, label))
    return corpus


class NaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.doc_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_words: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.total_docs = 0

    def train(self, corpus: Iterable[LabeledText]) -> None:
        for text, label in corpus:
            words = tokenize(text)
            if not words:
                continue

            self.doc_counts[label] += 1
            self.word_counts[label].update(words)
            self.total_words[label] += len(words)
            self.vocab.update(words)

        self.total_docs = sum(self.doc_counts.values())
        if self.total_docs == 0:
            raise ValueError("Training corpus is empty.")

    def predict(self, text: str) -> str:
        words = tokenize(text)
        if not words:
            raise ValueError("Cannot predict an empty text.")

        best_label = ""
        best_score = float("-inf")
        vocab_size = len(self.vocab)

        for label in self.doc_counts:
            score = math.log(self.doc_counts[label] / self.total_docs)
            denominator = self.total_words[label] + self.alpha * vocab_size

            for word in words:
                count = self.word_counts[label][word]
                likelihood = (count + self.alpha) / denominator
                score += math.log(likelihood)

            if score > best_score:
                best_label = label
                best_score = score

        return best_label

    def evaluate(self, corpus: Iterable[LabeledText]) -> float:
        correct = 0
        total = 0
        for text, label in corpus:
            correct += self.predict(text) == label
            total += 1
        if total == 0:
            raise ValueError("Evaluation corpus is empty.")
        return correct / total


DEMO_TRAIN: list[LabeledText] = [
    ("这 部 电影 很 精彩 演员 表演 真 好", "positive"),
    ("剧情 好看 节奏 很 棒", "positive"),
    ("我 喜欢 这个 故事 很 温暖", "positive"),
    ("电影 太 糟糕 剧情 无聊", "negative"),
    ("演员 表演 很 差 我 不 喜欢", "negative"),
    ("节奏 拖沓 故事 无聊", "negative"),
]

DEMO_TEST: list[LabeledText] = [
    ("这个 故事 很 精彩", "positive"),
    ("剧情 无聊 表演 很 差", "negative"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multinomial Naive Bayes text classifier demo.")
    parser.add_argument("--train", type=Path, help="Training TSV file: label<TAB>text")
    parser.add_argument("--test", type=Path, help="Testing TSV file: label<TAB>text")
    args = parser.parse_args()

    train_corpus = load_tsv(args.train) if args.train else DEMO_TRAIN
    test_corpus = load_tsv(args.test) if args.test else DEMO_TEST

    nb = NaiveBayes()
    print("开始训练...")
    nb.train(train_corpus)
    print(f"词表大小：{len(nb.vocab)}")
    print(f"类别统计：{dict(nb.doc_counts)}")
    print(f"准确率：{nb.evaluate(test_corpus):.2%}")

    samples = ["这 部 电影 很 好看", "剧情 拖沓 很 无聊"]
    for sample in samples:
        print(f"预测：{sample} -> {nb.predict(sample)}")


if __name__ == "__main__":
    main()
