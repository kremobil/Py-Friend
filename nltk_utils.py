import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence: str) -> list:
    return nltk.word_tokenize(sentence)
def stem(word: str) -> str:
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence: list, all_words: list) -> list:
    pass

a = "How long does shiping take?"
print(a)
a = tokenize(a)
print(a)
b = ["organize", "organizes", "organizing"]
stemmed_a = [stem(w) for w in a]
stemmed_b = [stem(w) for w in b]
print(stemmed_a, stemmed_b)