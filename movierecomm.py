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

# Load data
@st.cache
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

# Function to filter movies based on search query
def filter_movies(movies, search_query):
    return movies[movies['title'].str.contains(search_query, case=False)]

# Function to display movie recommendations
def display_recommendations(movies, ratings, selected_movies):
    st.write('## Recommended Movies')
    for movie_id in selected_movies:
        movie_title = movies.loc[movies['movieId'] == movie_id, 'title'].iloc[0]
        st.write('- ' + movie_title)

# Streamlit UI
st.title('Movie Recommendation System')

# Load data
movies, ratings = load_data()

# Search bar for user to search for movie names
search_query = st.text_input('Search for a movie')

# Filter movies based on search query
filtered_movies = filter_movies(movies, search_query)

# Scroll bar for user to choose movies
selected_movies = st.multiselect('Choose movies', filtered_movies['movieId'])

# Display recommended movies
if st.button('Get Recommendations'):
    display_recommendations(movies, ratings, selected_movies)
