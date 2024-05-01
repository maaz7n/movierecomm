import streamlit as st
import pandas as pd
import base64

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

# Encode background image as base64 string
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set background image (replace 'movie_poster.jpg' with actual file path)
background_image = get_base64_of_bin_file("movie.jpg")
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url('data:image/png;base64,{background_image}') center center no-repeat;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Movie Recommendation System')

st.markdown("""
    Welcome to the Movie Recommendation System! Select a movie from the dropdown menu 
    and click the button to get recommendations based on similar genres.
""")

# Sidebar with movie selection
st.sidebar.header('Select a Movie')
selected_movie = st.sidebar.selectbox(
    'Choose a movie:',
    movies_df['title'].values
)

# Main content area for recommendations
if st.sidebar.button('Get Recommendations'):
    st.subheader(f'Recommendations for **{selected_movie}**')
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")
    else:
        st.warning("No recommendations found for this movie.")
