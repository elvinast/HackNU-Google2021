import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

X = pickle.load(open("X.pickle", "rb"))
tfidf = pickle.load(open("tfidf.pickle", "rb"))
vector = pickle.load(open("vector.pickle", "rb"))

def search(tfidf_matrix, model, request, top_n=5):
    request_transform = model.transform([request])
    similarity = np.dot(request_transform, np.transpose(tfidf_matrix))
    x = np.array(similarity.toarray()[0])
    indices = np.argsort(x)[-5:][::-1]
    return indices

def print_result(request_content, indices, X):
    print('Search : ' + request_content)
    print('Best Results :')
    for i in indices:
        print('id = {0} ; title - {1}'.format(i, X['title'].loc[i]))

print('Ready')
while True:
    s = input()
    result = search(tfidf, vector, s, top_n=5)
    print(result)
    print_result(s, result, X)