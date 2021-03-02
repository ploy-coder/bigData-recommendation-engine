
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv("booksnew.csv")
#print df.columns
##Step 2: Select Features
features = ['genre','subgenre']

##Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('') #filling all NaNs with blank string

def combine_features(row):

        return row['genre']+" "+row['subgenre']



df["combined_features"] = df.apply(combine_features,axis=1)
#print "Combined Features:", df["combined_features"].head()

##Step 4: Create count matrix from this new combined column


cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
book_user_likes = "Amulet of Samarkand, The"

## Step 6: Get index of this book from its title
book_index = get_index_from_title(book_user_likes)
similar_book = list(enumerate(cosine_sim[book_index]))

## Step 7: Get a list of similar book in descending order of similarity score
sorted_similar_book = sorted(similar_book,key=lambda x:x[1],reverse=True)[1:]



## Step 8: Print film of first 50 book
i=0
print("Top 5 similar book to "+book_user_likes+" are:\n")
for element in sorted_similar_book:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>= 5:
        break