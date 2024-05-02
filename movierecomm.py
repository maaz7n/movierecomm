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

# Streamlit UI
st.title('Movie Recommendation System')

# Load data
movies, ratings = load_data()

# Scroll bar for user to choose movies
selected_movies = st.multiselect('Choose movies', movies['title'])

# Display recommended movies
if st.button('Get Recommendations'):
    st.write('## Recommended Movies')
    for movie_title in selected_movies:
        st.write('- ' + movie_title)
