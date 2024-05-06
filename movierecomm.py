# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

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
similarity_matrix = compute_similarity_matrix(movies_df)

# Streamlit UI
st.title("Movie Recommendation System")

# User input
movie_title = st.text_input("Enter a movie title: ")

# Generate recommendations
if movie_title.strip() != "":
    # Find the index of the movie in the dataset
    movie_index = movies_df[movies_df["genres"] == movie_title].index[0]


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
