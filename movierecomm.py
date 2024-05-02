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

# Load ratings and movies data
@st.cache
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

# Function to get movie recommendations
def get_recommendations(movie_title, ratings, movies, n=10):
    # Get movie ID for the given movie title
    movie_id = movies[movies['title'].str.contains(movie_title, case=False)]['movieId'].values
    if len(movie_id) == 0:
        return None
    # Get similar movies
    similar_movies = pd.DataFrame(columns=['movieId', 'similarity'])
    for m_id in movie_id:
        # Find movies similar to the searched movie
        similar = ratings[ratings['movieId'] == m_id].merge(ratings, on='userId')
        if not similar.empty:
            similar = similar.groupby('movieId_y').apply(lambda x: (x['rating_x'] * x['rating_y']).sum() / (x['rating_x'].pow(2).sum() ** 0.5 * x['rating_y'].pow(2).sum() ** 0.5)).reset_index(name='similarity')
            similar_movies = similar_movies.append(similar, ignore_index=True)
    # Get top N recommended movies
    if not similar_movies.empty:
        similar_movies = similar_movies.groupby('movieId').mean().reset_index().sort_values(by='similarity', ascending=False)
        top_movie_ids = similar_movies.head(n)['movieId']
        top_movies = movies[movies['movieId'].isin(top_movie_ids)]['title']
        return top_movies
    else:
        return None

# Streamlit UI
st.title('Movie Recommendation System')

# Load ratings and movies data
ratings, movies = load_data()

# Input for user to search for a movie
search_query = st.text_input('Search for a movie')

# Button to trigger recommendation
if st.button('Get Recommendations'):
    if search_query:
        # Get movie recommendations for the searched movie
        recommended_movies = get_recommendations(search_query, ratings, movies)
        if recommended_movies is not None:
            # Display recommended movies
            st.write('## Recommended Movies')
            for movie in recommended_movies:
                st.write('- ' + movie)
        else:
            st.write('No recommendations found for the searched movie.')
    else:
        st.write('Please enter a movie title to get recommendations.')
