import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

A = pd.read_csv("Desktop/NLP/archive/articles1.csv")
B = pd.read_csv("Desktop/NLP/archive/articles2.csv")
C = pd.read_csv("Desktop/NLP/archive/articles3.csv")
X = A + B + C
X = X[pd.isna(X['content']) == False]
contents = X['content']

vector = TfidfVectorizer(max_df=0.3)
tfidf = vector.fit_transform(contents)

pickle.dump(X, open("X.pickle", "wb"))
pickle.dump(tfidf, open("tfidf.pickle", "wb"))
pickle.dump(vector, open("vector.pickle", "wb"))
print('Done')