import streamlit as st
import pandas as pd

# Set page background to a URL
st.markdown(
    """
    <style>
    body {
        background-image: url("https://raw.githubusercontent.com/maaz7n/movierecomm/main/background.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load movie ratings and movie data
@st.cache
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# Function to get movie recommendations
def get_recommendations(user_id, ratings, movies, n=10):
    # Get movies rated by the user
    user_ratings = ratings[ratings['userId'] == user_id]
    user_rated_movie_ids = user_ratings['movieId'].tolist()
    # Calculate average rating of user
    avg_rating = user_ratings['rating'].mean()
    # Get similar movies
    similar_movies = pd.DataFrame(columns=['movieId', 'similarity'])
    for movie_id in user_rated_movie_ids:
        # Find movies similar to the rated movie
        similar = ratings[ratings['movieId'] == movie_id].merge(ratings, on='userId')
        similar = similar[similar['movieId_y'] != movie_id]
        similar = similar.groupby('movieId_y').apply(lambda x: (x['rating_x'] * x['rating_y']).sum() / (x['rating_x'].pow(2).sum() ** 0.5 * x['rating_y'].pow(2).sum() ** 0.5)).reset_index(name='similarity')
        similar_movies = similar_movies.append(similar, ignore_index=True)
    # Filter out movies already rated by the user
    similar_movies = similar_movies[~similar_movies['movieId'].isin(user_rated_movie_ids)]
    # Get top N recommended movies
    similar_movies = similar_movies.groupby('movieId').mean().reset_index().sort_values(by='similarity', ascending=False)
    top_movie_ids = similar_movies.head(n)['movieId']
    top_movies = movies[movies['movieId'].isin(top_movie_ids)]['title']
    return top_movies

# Streamlit UI
st.title('Movie Recommendation System')

# Input for user to enter user ID
user_id = st.number_input('Enter your user ID', min_value=1, max_value=610, value=1, step=1)

# Button to trigger recommendation
if st.button('Get Recommendations'):
    # Get movie recommendations for the user
    recommended_movies = get_recommendations(user_id, ratings, movies)
    # Display recommended movies
    st.write('## Recommended Movies')
    for movie in recommended_movies:
        st.write('- ' + movie)
