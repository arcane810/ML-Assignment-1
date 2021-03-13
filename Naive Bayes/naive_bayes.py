# importing libraries
import pandas as pd
import numpy as np
import re
import math

# importing the dataset
df = pd.read_csv('a1_d3.txt', sep = '\t', header = None, names = ['review', 'sentiment'])
df = df.sample(frac=1).reset_index(drop=True)

# list of stop words
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# text pre-processing
def word_extraction(sentence):       
    words = re.sub("[^\w]", " ",  sentence).split()    
    cleaned_text = [w.lower() for w in words if w not in stop_words and w.isalpha()]         
    return cleaned_text

# generating vocabulary
def tokenize(sentences):    
    words = []    
    for sentence in sentences:        
        w = word_extraction(sentence)        
        words.extend(w)            
        words = sorted(list(set(words)))    
    return words

# generating equivalent vector for a sentence
def vectorize(words, vocab):
    bag_vector = np.zeros(len(vocab), int) 
    for w in words:            
        for i,word in enumerate(vocab):                
            if word == w:                     
                bag_vector[i] = 1
    return bag_vector

# generating bag of words representation of all sentences
def generate_bow(allsentences):        
    vocab = tokenize(allsentences)    
    vectors = np.zeros(len(vocab), int)
    for sentence in allsentences:        
        words = word_extraction(sentence)        
        bag_vector = vectorize(words, vocab)               
        vectors = np.vstack((vectors,bag_vector))
       
    vectors = np.delete(vectors, (0), axis=0)

    return vocab, vectors

#calculating conditional prior probability for each word (with Laplace Smoothing)
def prob_list(Y, vectors, vocab):
    a = np.zeros(len(vocab)) 
    for i,w in enumerate(vocab):
        nx = (vectors[:, i] * Y).sum()
        n = Y.sum()
        p = (nx + 1) / (n + 2)
        a[i] = p
    return a

# calculating log posterior probability
def prob(X, Y, p_list, vocab):
    P = math.log(Y.sum()) - math.log(len(Y))
    for i,w in enumerate(vocab):
        p = p_list[i]
        if(X[i] == 0):
            P += math.log(1-p)
        else:

            P += math.log(p)
    return P

#5-fold cross validation
A = np.empty(0, dtype = int)
F = np.empty(0, dtype = int)
for i in range(0,5):
    train = df.loc[[x for x in range(0,1000) if x not in range (i*200,(i+1)*200)]]
    test = df.loc[[x for x in range(0,1000) if x in range (i*200,(i+1)*200)]]
    vocab, vectors = generate_bow(train['review'].values)
    Y = train['sentiment'].values
    res = np.empty(0, dtype = int)
    Y1 = test['sentiment'].values
    prob_pos = prob_list(Y, vectors, vocab)
    prob_neg = prob_list(1-Y, vectors, vocab)

    for sentence in test['review'].values:
        words = word_extraction(sentence)       
        X = vectorize(words, vocab)
        pos = prob(X, Y, prob_pos, vocab)
        neg = prob(X, 1-Y, prob_neg, vocab)
        if(pos > neg):
            res = np.append(res, 1)
        else:
            res = np.append(res, 0)
    
    correct = ((res == Y1) * 1).sum()
    total = len(res)
    accuracy = correct / total;
    A = np.append(A, accuracy)

    precision = (res*Y1).sum()/res.sum()
    recall = (res*Y1).sum()/Y1.sum()
    fscore = (2 * precision * recall)/(precision + recall)
    F = np.append(F, fscore)

    print('Test fold', i + 1, ':', 'Accuracy =', '%.2f' % accuracy, 'F-score =', '%.2f' % fscore)

# Accuracy
A_mean = np.mean(A)
A_std = np.std(A)

print ('Accuracy = ', '%.2f' % A_mean, '±', '%.2f' % A_std)

# F-score

F_mean = np.mean(F)
F_std = np.std(F)

print ('F-score = ', '%.2f' % F_mean, '±', '%.2f' % F_std)
