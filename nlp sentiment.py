# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:55:39 2023

@author: donald
"""

import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load movie reviews dataset
nltk.download('movie_reviews')
reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# view the movie reviews dataset in a df
df = pd.DataFrame(reviews, columns=['text','label'])
df.head()


# Split the data into training and testing sets
X = [review for review, label in reviews]
y = [label for review, label in reviews]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectors, y_train)

# Evaluate the classifier on the testing set
accuracy = clf.score(X_test_vectors, y_test)
print("Accuracy:", accuracy)


# Use the model...................................

# Load text data to predict the sentiment of
new_text = ["This movie was fantastic!", "I hated this book so much.", "this really sucked", "i really don't know what to think", "could be better", "couldn't be better"]

# Vectorize the new text data 
new_text_vectors = vectorizer.transform(new_text)

# Make predictions 
predictions = clf.predict(new_text_vectors)

# Print the predictions
for i, prediction in enumerate(predictions):
    print("Text:", new_text[i])
    print("Sentiment:", prediction)


