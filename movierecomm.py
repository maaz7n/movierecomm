import streamlit as st
import pandas as pd

# Load movies and ratings data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Function to perform collaborative filtering
def collaborative_filtering(user_id, n=10):
    # Filter ratings for the given user
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # Merge user ratings with movie data
    user_movie_ratings = pd.merge(user_ratings, movies, on='movieId')
    
    # Sort movies by rating in descending order
    user_movie_ratings = user_movie_ratings.sort_values(by='rating', ascending=False)
    
    # Get top N recommended movies
    top_movies = user_movie_ratings.head(n)
    
    return top_movies['title']

# Streamlit UI
st.title('Movie Recommendation System')

# User input for user ID
user_id = st.number_input('Enter User ID', min_value=1, max_value=1000)

# Button to trigger recommendation
if st.button('Get Recommendations'):
    # Perform collaborative filtering to get movie recommendations
    recommended_movies = collaborative_filtering(user_id)
    
    # Display recommended movies
    st.write('## Recommended Movies')
    for movie in recommended_movies:
        st.write('- ' + movie)
