# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

# Function to calculate similarity based on genres using sklearn
def calculate_similarity(movie_genres_1, movie_genres_2):
    if not movie_genres_1 or not movie_genres_2:
        return 0  # If any of the genres is empty, return 0 similarity
    try:
        genres_1 = set(movie_genres_1.split('|'))
        genres_2 = set(movie_genres_2.split('|'))
    except AttributeError:
        return 0  # If genres are not in the expected format, return 0 similarity
    # Create binary vectors representing presence/absence of genres
    vectorizer = CountVectorizer(binary=True)
    vectorizer.fit(list(genres_1.union(genres_2)))
    vec_1 = vectorizer.transform([movie_genres_1])
    vec_2 = vectorizer.transform([movie_genres_2])
    # Compute cosine similarity between the vectors
    similarity_matrix = cosine_similarity(vec_1, vec_2)
    similarity = similarity_matrix[0, 0]
    return similarity

similarity_matrix = compute_similarity_matrix(movies_df)

# Streamlit UI
st.title("Movie Recommendation System")

# User input
movie_title = st.text_input("Enter a movie title: ")

# Generate recommendations
if movie_title.strip() != "":
    # Find the index of the movie in the dataset
    movie_index = movies_df[movies_df["movie_title"] == movie_title].index[0]

    # Get similarity scores for the movie
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort the movies based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Display top 5 similar movies
    st.subheader("Top 5 Recommended Movies:")
    for i in range(5):
        recommended_movie_index = similarity_scores[i][0]
        recommended_movie_title = movies_df.iloc[recommended_movie_index]["movie_title"]
        st.write(f"{i+1}. {recommended_movie_title}")
