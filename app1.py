import streamlit as st
import pandas as pd

import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import requests



#print(os.getcwd())
st.title("Reccomendation System")

df=pickle.load(open("df.pkl","rb"))

df=pd.DataFrame(df)

new=pickle.load(open("Movies.pkl","rb"))


new=pd.DataFrame(new)


movie_list= new["title"]

movie_name_list=movie_list.values

cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

similarity = cosine_similarity(vector)


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    reccomended_movies=[]
    
    for i in distances[1:6]:
        movie_id=i[0]
        reccomended_movies.append(new.iloc[i[0]].title)
    return reccomended_movies


selected_movie_name=st.selectbox("select the movie",movie_name_list)

if st.button("Reccomend"):
    for i in (recommend(selected_movie_name)):
        st.write(i)
        if st.button(i):
            st.write(i)
