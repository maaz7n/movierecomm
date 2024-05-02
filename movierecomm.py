import streamlit as st
import pandas as pd

# Set page background to a URL
st.markdown(
    """
    <style>
    body {
        background-image: url("YOUR_BACKGROUND_IMAGE_URL");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load movies and ratings data
@st.cache
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

# Function to perform collaborative filtering
def collaborative_filtering(movies, ratings, n=10):
    # Group ratings by movie and count number of ratings for each movie
    movie_ratings_count = ratings.groupby('movieId').size().reset_index(name='rating_count')
    # Sort movies by rating count in descending order
    popular_movies = movie_ratings_count.sort_values(by='rating_count', ascending=False)
    # Get top N recommended movie IDs
    top_movie_ids = popular_movies.head(n)['movieId']
    # Get movie titles corresponding to top movie IDs
    top_movies = movies[movies['movieId'].isin(top_movie_ids)]['title']
    return top_movies

# Streamlit UI
st.title('Movie Recommendation System')

# Load movies and ratings data
movies, ratings = load_data()

# Search bar for user to search for movie names
search_query = st.text_input('Search for a movie')

# Filter movies based on search query
filtered_movies = movies[movies['title'].str.contains(search_query, case=False)]
filtered_movies = filtered_movies[filtered_movies['movieId'].isin(ratings['movieId'])]

# Display filtered movies
st.write('## Filtered Movies')
for movie in filtered_movies['title']:
    st.write('- ' + movie)
