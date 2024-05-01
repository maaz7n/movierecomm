import streamlit as st
import pandas as pd
import difflib

# Load data
@st.cache
def load_data():
    return pd.read_csv("/Users/mohammedmazin/Downloads/movieRecomm.py")  # Update with your file path

movies_data = load_data()

# Preprocess data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
movies_data['combined_features'] = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Recommendation function
def recommend_movies(movie_name):
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'].tolist(), n=1)
    close_match = find_close_match[0] if find_close_match else None
    if close_match:
        similar_movies = movies_data[movies_data['title'] == close_match]['combined_features']
        similar_movies = similar_movies.apply(lambda x: x.split())
        similar_movies = similar_movies.explode().value_counts().index.tolist()
        return similar_movies
    else:
        return []

# Streamlit app
st.title('Movie Recommendation System')

# User input
movie_name = st.text_input('Enter your favorite movie name:')
if movie_name:
    recommendations = recommend_movies(movie_name)
    if recommendations:
        st.subheader('Recommended Movies:')
        for i, movie in enumerate(recommendations[:10], 1):
            st.write(f"{i}. {movie}")
    else:
        st.write('No close match found for the entered movie name.')
