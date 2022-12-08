import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from numpy import ndarray

stemmer = PorterStemmer()


def tokenize(sentence: str) -> list:
    return nltk.word_tokenize(sentence)


def stem(word: str) -> str:
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: list, all_words: list) -> ndarray:
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0

    return bag


sentence = ["hello", "world", "how", "are", "you", "today"]
words = ["hello", "world", "how", "are", "green", "you", "today", "stem"]
bag = bag_of_words(sentence, words)
print(bag)
