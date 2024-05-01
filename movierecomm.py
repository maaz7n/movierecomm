import streamlit as st
import pandas as pd

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

# Function to calculate similarity based on genres
def calculate_similarity(movie_genres_1, movie_genres_2):
    if not movie_genres_1 or not movie_genres_2:
        return 0  # If any of the genres is empty, return 0 similarity
    try:
        genres_1 = set(movie_genres_1.split('|'))
        genres_2 = set(movie_genres_2.split('|'))
    except AttributeError:
        return 0  # If genres are not in the expected format, return 0 similarity
    intersection = genres_1.intersection(genres_2)
    similarity = len(intersection) / (len(genres_1) + len(genres_2) - len(intersection))
    return similarity

# Function to get movie recommendations
def get_recommendations(movie_title, threshold=0.2):
    movie_row = movies_df[movies_df['title'] == movie_title]
    movie_genres = movie_row['genres'].values[0]
    recommendations = []
    for index, row in movies_df.iterrows():
        if row['title'] != movie_title:
            similarity = calculate_similarity(movie_genres, row['genres'])
            if similarity >= threshold:
                recommendations.append(row['title'])
    return recommendations

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #1e272e;
        border-radius: 10px;
        padding: 20px;
    }
    .sidebar .sidebar-content .stButton {
        background-color: #7ed6df;
        color: #ffffff;
        border-radius: 5px;
        padding: 0.375rem 0.75rem;
    }
    .sidebar .sidebar-content .stButton:hover {
        background-color: #3a3e4b;
    }
    .main .block-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """
)

# Sidebar
with st.sidebar:
    st.markdown("## Sidebar")
    selected_movie = st.selectbox('Choose a movie:', movies_df['title'].values, index=0)
    st.markdown("---")
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(selected_movie)
        if recommendations:
            st.markdown("### Recommendations")
            for i, movie in enumerate(recommendations):
                st.write(f"{i+1}. {movie}")
        else:
            st.warning("No recommendations found for this movie.")

# Main content
st.title('Movie Recommendation System')
st.markdown("---")
st.markdown("Welcome to the Movie Recommendation System! Select a movie from the sidebar and click the button to get recommendations based on similar genres.")
