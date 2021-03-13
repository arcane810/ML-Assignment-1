import numpy as np
import matplotlib.pyplot as plt

stopwords = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]

punctuation_symbols = [".", ",", "!", "/", ":", "-", ";", "(", ")"]


def preprocessing():
    dataset_x = []
    dataset_y = []
    with open("./dataset_NB.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            new_line = ""
            for i in range(len(line)):
                if line[i] in punctuation_symbols:
                    new_line += " "
                else:
                    new_line += line[i]
            words = new_line.split()
            dataset_x.append(
                [
                    word.lower()
                    for word in words[:-1]
                    if (not word.isnumeric() and word.lower() not in stopwords)
                ]
            )
            dataset_y.append(words[-1])
    return dataset_x, dataset_y


dataset_x, dataset_y = preprocessing()


def get_probabilities(dataset_x, dataset_y, alpha=1):
    vocab = list(set([word for words in dataset_x for word in words]))
    vocab.sort()
    index = {}
    i = 0
    for word in vocab:
        index[word] = i
        i += 1
    pclass1 = np.zeros(len(vocab))
    pclass2 = np.zeros(len(vocab))
    p1sum = 0
    p2sum = 0
    for i in range(len(dataset_y)):
        if dataset_y[i] == "1":
            p1sum += len(dataset_x[i])
            for word in dataset_x[i]:
                pclass1[index[word]] += 1
        else:
            p2sum += len(dataset_x[i])
            for word in dataset_x[i]:
                pclass2[index[word]] += 1

    pwc1 = (pclass1+alpha)/(p1sum+alpha*len(vocab))
    pwc2 = (pclass2+alpha)/(p2sum+alpha*len(vocab))
    post_c1 = p1sum/(p1sum+p2sum)
    post_c2 = p2sum/(p1sum+p2sum)
    return pwc1, pwc2, p1sum, p2sum, index


naive_bayes(dataset_x, dataset_y)