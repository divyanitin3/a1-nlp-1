import random
from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))

    def get_vocab_size(self):
        return len(self.indexer)

    def extract_features(self, sentence: List[str]) -> Counter:
        feat_vector = Counter()

        for word in sentence:
            if word not in self.stop_words:
                word_idx = self.indexer.add_and_get_index(f"Unigram={word}")
                feat_vector[word_idx] += 1
        
        return feat_vector

class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))
        
    def get_vocab_size(self):
        return len(self.indexer)

    def extract_features(self, sentence):
        feature_vector = Counter()
        
        # we remove the stop words from the sentence
        filtered_sentence = [word for word in sentence if word not in self.stop_words]
        
        for i in range(len(filtered_sentence) - 1):
            bigram = f"{filtered_sentence[i]}_{filtered_sentence[i+1]}"
            bigram_index = self.indexer.get_index(f"Bigram={bigram}")
            feature_vector[bigram_index] += 1
        
        return feature_vector

class BetterFeatureExtractor(FeatureExtractor):
    """
    A better feature extractor that combines both unigram and bigram features.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))

    def get_vocab_size(self):
        return len(self.indexer)

    def extract_features(self, sentence):
        """
        Extracts both unigram and bigram features from a sentence.
        :param sentence: List of words (strings)
        :return: A dictionary where keys are indices of features and values are counts.
        """
        feature_vector = Counter()
         # we remove the stop words from the sentence
        filtered_sentence = [word for word in sentence if word not in self.stop_words]

        for word in filtered_sentence:
            unigram_index = self.indexer.add_and_get_index(f"Unigram={word}")
            feature_vector[unigram_index] += 1

        for i in range(len(filtered_sentence) - 1):
            bigram = f"{filtered_sentence[i]}_{filtered_sentence[i+1]}"
            bigram_index = self.indexer.add_and_get_index(f"Bigram={bigram}")
            feature_vector[bigram_index] += 1
        
        return feature_vector

class SentimentClassifier(object):
    def predict(self, sentence: List[str]) -> int:
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    def __init__(self):
        super().__init__(None)

    def predict(self, sentence: List[str]) -> int:
        return 1

class PerceptronClassifier(SentimentClassifier):
    def __init__(self, feature_extractor: FeatureExtractor, vocab_size, epochs=18, learning_rate=0.2):
        self.feature_extractor = feature_extractor
        self.weights = np.zeros(vocab_size) 
        self.bias = 0.0
        self.epochs = epochs
        self.learning_rate = learning_rate

    def calc_learning_rate(self, epoch):
        return self.initial_learning_rate / (epoch + 1)

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence)
        score = 0.0
        for feature, value in features.items():
            if feature < len(self.weights):
                score += self.weights[feature] * value
        return 1 if score > 0 else 0
    
    def train(self, training_examples):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            random.shuffle(training_examples)

            for eg in training_examples:
                features = self.feature_extractor.extract_features(eg.words)
                correct_label = eg.label
                predicted = self.predict(eg.words)

                if predicted != correct_label:
                    update = 1 if correct_label == 1 else -1

                    for index, value in features.items():
                        self.weights[index] += self.learning_rate * update * value

class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, feature_extractor, vocab_size, epochs=20, learning_rate=0.2):
        self.feature_extractor = feature_extractor
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(vocab_size)
        self.bias = 0.0

    def sigmoid(self, score: float) -> float:
        return 1 / (1 + np.exp(-score))

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence)
        score = 0.0

        for feature, value in features.items():
            if feature < len(self.weights):
                score += self.weights[feature] * value

        probability = self.sigmoid(score)
        return 1 if probability >= 0.5 else 0

    def update_weights(self, features, label, probability, learning_rate):
        error = label - probability
        for feature, value in features.items():
            if feature < len(self.weights):
                #regularize
                self.weights[feature] += learning_rate * (error * value - self.reg_strength * self.weights[feature])

    def compute_loss_and_gradients(self, example):
        features = self.feature_extractor.extract_features(example.words)
        correct_label = example.label
        score = self.bias

        for feat, value in features.items():
            if feat < len(self.weights):
                score += self.weights[feat] * value   

        probability = self.sigmoid(score)
        loss =- (correct_label * np.log(probability) + (1 - correct_label) * np.log(1 - probability))
        error = probability - correct_label
        gradients_weight = np.zeros_like(self.weights)

        for feat, value in features.items():
            if feat < len(self.weights):
                gradients_weight[feat] = error * value

        gradient_bias = error

        return loss, gradients_weight, gradient_bias
    
    def train(self, train_exs):
        for epoch in range(self.epochs):
            loss_total = 0.0
            random.shuffle(train_exs)

            for eg in train_exs:
                loss, gradients_weight, gradient_bias = self.compute_loss_and_gradients(eg)
                loss_total += loss
                self.weights -= self.learning_rate * gradients_weight
                self.bias -= self.learning_rate * gradient_bias     
                       
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss_total:.4f}")

def train_perceptron(train_examples: List[SentimentExample], feature_extractor: FeatureExtractor) -> PerceptronClassifier:
    for eg in train_examples:
        feature_extractor.extract_features(eg.words)

    vocab_size = feature_extractor.get_vocab_size()
    classifier = PerceptronClassifier(feature_extractor, vocab_size)
    classifier.train(train_examples)

    return classifier

def train_logistic_regression(train_examples: List[SentimentExample], feature_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    for eg in train_examples:
        feature_extractor.extract_features(eg.words)

    vocab_size = feature_extractor.get_vocab_size()

    # Initialize the classifier
    classifier = LogisticRegressionClassifier(feature_extractor, vocab_size)
    classifier.train(train_examples)

    return classifier


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model