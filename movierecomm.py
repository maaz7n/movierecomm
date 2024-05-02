import streamlit as st
import pandas as pd

# Set page background to a URL
st.markdown(
    """
    <style>
    body {
        background-image: url('https://raw.githubusercontent.com/maaz7n/movierecomm/main/background.jpg');
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
def collaborative_filtering(movies, ratings, min_rating=3, n=10):
    # Filter movies by minimum rating
    filtered_ratings = ratings[ratings['rating'] >= min_rating]
    # Group ratings by movie and count number of ratings for each movie
    movie_ratings_count = filtered_ratings.groupby('movieId').size().reset_index(name='rating_count')
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

# Scroll bar for user to select minimum rating
min_rating = st.sidebar.slider('Select minimum rating', min_value=1, max_value=5, value=3, step=1)

# Button to trigger recommendation
if st.sidebar.button('Get Recommendations'):
    # Perform collaborative filtering to get movie recommendations
    recommended_movies = collaborative_filtering(movies, ratings, min_rating)
    # Display recommended movies
    st.write('## Recommended Movies')
    for movie in recommended_movies:
        st.write('- ' + movie)
