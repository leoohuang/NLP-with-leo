import math
from collections import Counter
from tokenizer import tokenize
from corpus import CategorizedCorpus

class NaiveBayes:
    def train(self, train_corpus):
        self.doc_counts = {}
        self.word_counts = {}
        self.vocab = set()
        for sent, cat in train_corpus:
            words = set(tokenize(sent))
            if cat not in self.doc_counts:
                self.doc_counts[cat] = 0
                self.word_counts[cat] = Counter()
            self.word_counts[cat].update(words)
            self.doc_counts[cat] +=1
            self.vocab.update(words)
        self.total_docs = sum(self.doc_counts.values())
        self.V = len(self.vocab)
    
    def predict(self, text):
        words = set(tokenize(text))
        best_cat = None
        best_socre = float('-inf')
        for cat in self.word_counts:
            score = math.log(self.doc_counts[cat]/self.total_docs)
            total = sum(self.word_counts[cat].values())
            for word in words:
                count = self.word_counts[cat][word]
                score += math.log((count + 1)/(total+self.V))
            if score > best_socre:
                best_socre = score
                best_cat = cat
        return best_cat
    
    def evaluate(self, test_corpus):
        correct = 0
        total = 0
        for text, cat in test_corpus:
            if self.predict(text) == cat:
                correct += 1
            total += 1
        return correct / total 
    

if __name__ == "__main__":
    train_path = "/Users/leohuang/Downloads/NLP/train"
    test_path = "/Users/leohuang/Downloads/NLP/test"
    nb = NaiveBayes()
    train_corpus = CategorizedCorpus(train_path)
    test_corpus = CategorizedCorpus(test_path)
    print(f"开始训练...")
    nb.train(train_corpus)
    acc = nb.evaluate(test_corpus)
    print(f"准确率：{acc}")