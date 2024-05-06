import streamlit as st
import pandas as pd
import base64
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

# Function to compute similarity matrix based on genres
def compute_similarity_matrix(data):
    try:
        # Extract genres from the "genres" column
        genres_list = data['genres'].tolist()

        # Create binary vectors representing presence/absence of genres
        vectorizer = CountVectorizer(binary=True)
        genre_matrix = vectorizer.fit_transform(genres_list)

        # Compute cosine similarity between genre vectors
        similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
        return similarity_matrix
    except Exception as e:
        print("An error occurred while computing similarity matrix:", e)
        return None

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

# Function to compute similarity matrix based on genres
def compute_similarity_matrix(data):
    try:
        # Extract genres from the "genres" column
        genres_list = data['genres'].tolist()

        # Create binary vectors representing presence/absence of genres
        vectorizer = CountVectorizer(binary=True)
        genre_matrix = vectorizer.fit_transform(genres_list)

        # Compute cosine similarity between genre vectors
        similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
        
        # Debugging: Print shape of the similarity matrix
        print("Shape of similarity matrix:", similarity_matrix.shape)
        
        return similarity_matrix
    except Exception as e:
        print("An error occurred while computing similarity matrix:", e)
        return None



# Function to convert image to base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image from URL
def set_background_image(url):
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("%s");
        background-size: cover;
    }
    </style>
    ''' % url
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Main function
def main():
    # Load data
    movies_df = load_data()

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(movies_df)

    # Set background image from URL
    background_image_url = "https://raw.githubusercontent.com/maaz7n/movierecomm/main/background.jpg" # Replace with your URL
    set_background_image(background_image_url)

    # Streamlit UI
    st.title('Movie Recommendation System')

    # Select a movie
    selected_movie = st.selectbox('Select a movie:', movies_df['title'].values)

    # Get recommendations
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(selected_movie, movies_df, similarity_matrix)
        if recommendations:
            st.write("### Recommendations")
            for movie in recommendations:
                st.write(f"- {movie}")
        else:
            st.write("No recommendations found for this movie.")

if __name__ == "__main__":
    main()
