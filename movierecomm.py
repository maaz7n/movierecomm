# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

# Compute similarity matrix
def compute_similarity_matrix(data):
    data_numeric = data.drop(columns=["movie_title"])
    similarity_matrix = cosine_similarity(data_numeric, data_numeric)
    return similarity_matrix

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
    for i in range(1, 6):
        recommended_movie_index = similarity_scores[i][0]
        recommended_movie_title = movies_df.iloc[recommended_movie_index]["movie_title"]
        st.write(f"{i}. {recommended_movie_title}")
