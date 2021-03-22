#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans


# In[61]:


X1 = pd.read_csv("./articles1.csv")
X2 = pd.read_csv("./articles2.csv")
X3 = pd.read_csv("./articles3.csv")

X = X1 + X2 + X3
# X = X[pd.isna(X['title'])==False] #check if it exists(not null)
X = X[pd.isna(X['content'])==False]


# In[82]:


# tfidf calculation
articleContect = X['content']
vc = TfidfVectorizer(max_df=0.3, stop_words='english')
# print(vc)
TF_idf = vc.fit_transform(articleContect)
# print(TF_idf)


# In[83]:


# Request function : search the top_n articles from a request ( request = string)
def search(TF_idf_matrix, model, request):
    requestTransform = model.transform([request])
#     print(request_transform)
    similarity = np.dot(requestTransform,np.transpose(TF_idf_matrix))
#     print(similarity)
    x = np.array(similarity.toarray()[0])
    indices = np.argsort(x)[-5:][::-1]
    return indices

# Print the result
def print_result(requestContent,indices,X):
    print('Key word: ' + requestContent)
    print('Top 5 best matches:')
    for i in indices:
        print('Article id: ' + str(i) + ', title: ' + str(X['title'].loc[i]))
        print('Article content: \n' + str(X['content'].loc[i]))


# In[84]:


# request = 'find'
#request = text_content[0]
s = input()
ans = search(TF_idf, vc, s)
print_result(s, ans, X)


# In[ ]:




