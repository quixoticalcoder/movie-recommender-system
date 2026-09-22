import numpy as np
import pandas as pd
import ast

# Read CSV files
movies = pd.read_csv('.venv/movie.csv')
credits = pd.read_csv('.venv/credits.csv')

# Merge DataFrames on 'title'
movies = movies.merge(credits, on='title')
print(movies.head())

# Select relevant columns
movies = movies[['id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

# Drop rows with missing values in important columns
movies = movies.dropna(subset=['genres', 'keywords', 'cast', 'crew'])

# Function to convert string representation of lists to actual lists
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except (ValueError, SyntaxError):
        return []

# Apply conversion function to 'genres' column
movies['genres'] = movies['genres'].apply(lambda x: convert(x) if pd.notnull(x) else [])


for index, row in movies.iterrows():
    print(f"Movie: {row['title']} - Genres: {row['genres']}")
#print(movies.head())
print(movies['keywords'].apply(convert))

def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
#print(movies.head())

# to get the director names from the crew
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew']=movies['crew'].apply(fetch_director)
print(movies.head())

movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
##print(movies['overview'])
#print(movies.head())
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x ])
##print(movies.head(20))

#tag column
movies['tags']= movies['overview']+ movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
#print(movies.head())

# new dataframe only havings tag column
new_df = movies[['id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

cv.get_feature_names_out()
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True, key = lambda x :x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')

import pickle
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movie.dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))